import json
import os.path
from os import PathLike
import os.path as osp

from tqdm import tqdm
from typing import Union, Literal, Optional, List, Tuple
from cesped.datamanager.relionStarDataset import ParticlesRelionStarDataset

import torch
import numpy as np
import pandas as pd
import starfile
from starstack.particlesStar import ParticlesStarSet

from cesped.constants import default_configs_dir, defaultBenchmarkDir

from cesped.zenodo.bechmarkUrls import NAME_PARTITION_TO_RECORID
from cesped.zenodo.downloadFromZenodo import download_record, getDoneFname

"""
This module implements the ParticlesDataset class. A Pytorch Dataset for dealing with Cryo-EM particles 
in the CESPED benchmark
"""


class ParticlesDataset(ParticlesRelionStarDataset):
    """
    ParticlesDataset: A Pytorch Dataset for dealing with Cryo-EM particles in the CESPED benchmark.<br>
    It can download data automatically

    ```python
    #Loads the halfset 0 for the benchmark entry named "TEST"
    ds = ParticlesDataset(targetName="TEST", halfset=0, benchmarkDir="/tmp/cryoSupervisedDataset/")
    ```
    and each particle can be acessed as usually
    ```python
    img, rotMat, xyShiftAngs, confidence, metadata = ds[0]
    ```
    <br>
    """

    def __init__(self, targetName: Union[PathLike, str],
                 halfset: Literal[0, 1],
                 benchmarkDir: str = defaultBenchmarkDir,
                 image_size: Optional[int] = None,
                 apply_perImg_normalization: bool = True,
                 ctf_correction: Literal["none", "phase_flip"] = "phase_flip",
                 image_size_factor_for_crop: float = 0.25,
                 ):
        """
        ##Builder

        Args:
            targetName (Union[PathLike, str]): The name of the target to use. It is also the basename of \
            the directory where the data is.
            halfset (Literal[0, 1]): The second parameter.
            benchmarkDir (str): The root directory where the datasets are downloaded.
            image_size (Optional[int]): The final size of the image (after cropping). If None, keep the original size
            apply_perImg_normalization (bool): Apply cryo-EM per-image normalization. I = (I-noiseMean)/noiseStd.
            ctf_correction (Literal[none, phase_flip]): phase_flip will correct amplitude inversion due to defocus
            image_size_factor_for_crop (float): Fraction of the image size to be cropped. Final size of the image \
            is origSize*(1-image_size_factor_for_crop). It is important because particles in cryo-EM tend to \
            be only 50% to 25% of the total area of the image.

        """



        assert halfset in [0, 1], f"Error, data halfset should be 0 or 1. Currently it is {halfset}"
        self.targetName = targetName
        self.halfset = halfset
        self.benchmarkDir = osp.expanduser(benchmarkDir)

        self._symmetry = None
        self._lock = torch.multiprocessing.RLock()
        self._augmenter = None


        if not self._is_avaible():
            self._download()
        super().__init__(self.starFname, rootDir=self.datadir, symmetry=self.symmetry, image_size=image_size,
                         perImg_normalization="noiseStats" if apply_perImg_normalization else "none",
                         ctf_correction=ctf_correction, image_size_factor_for_crop=image_size_factor_for_crop)

    @property
    def datadir(self):
        """ the directory where the target is stored"""
        return osp.join(self.benchmarkDir, self.targetName)

    @property
    def starFname(self):
        """ the particles star filename"""
        return osp.join(self.datadir, f"particles_{self.halfset}.star")

    @property
    def stackFname(self):
        """ the particles mrcs filename"""
        return osp.join(self.datadir, f"particles_{self.halfset}.mrcs")

    @property
    def symmetry(self):
        """The point symmetry of the dataset"""
        if not self._is_avaible():
            self._download()
        if self._symmetry is None:
            with open(osp.join(self.datadir, f"info_{self.halfset}.json")) as f:
                self._symmetry = json.load(f)["symmetry"].upper()
        return self._symmetry

    @property
    def sampling_rate(self):
        """The particle image sampling rate in A/pixels"""
        return self.particles.sampling_rate

    @classmethod
    def addNewEntryLocally(cls, starFname: Union[str, PathLike], particlesRootDir: Union[str, PathLike],
                           newTargetName: Union[str, PathLike], halfset: Literal[0, 1], symmetry:str,
                           benchmarkDir: Union[str, PathLike] = defaultBenchmarkDir):
        """

        Adds a new dataset to the local copy of the CESPED benchmark. It rearanges the starfile content and copies the
        stack of images. Notice that it duplicates your particle images.
        Args:
            starFname (Union[str, PathLike]): The star filename with the particles to be added to the local benchmark
            particlesRootDir (Union[str, PathLike]): The root directory that is referred in the starFname (e.g. Relion project dir).
            newTargetName (Union[str, PathLike]): The name of the target to use.
            halfset (Literal[0, 1]): The second parameter.
            symmetry (str): The point symmetry of the dataset
            benchmarkDir (str): The root directory where the datasets are downloaded.

        """

        stack = ParticlesStarSet(starFname=starFname, particlesDir=particlesRootDir)
        assert isinstance(stack[len(stack) - 1][0], np.ndarray), "Error, there is some problem reading your data"
        newTargetDir = os.path.join(benchmarkDir, newTargetName)
        os.makedirs(newTargetDir, exist_ok=True)
        newStarFname = os.path.join(newTargetDir, f"particles_{halfset}.star")
        newStackFname = os.path.join(newTargetDir, f"particles_{halfset}.mrcs")
        stack.save(newStarFname, newStackFname)

        with open(os.path.join(newTargetDir, f"info_{halfset}.json"), "w") as f: #TODO: move name template to download/uploadFromZenodo
            json.dump({"symmetry": symmetry.upper()}, f)

        with open(getDoneFname(newTargetDir, halfset), "w") as f:
            f.write("%s\n" % newTargetName)

    @classmethod
    def getCESPEDEntries(cls) -> List[Tuple[str, int]]:
        """
        Returns the list of available entries in benchmarkDir

        Returns:
            List[Tuple[str,int]]: the list of available entries in benchmarkDir

        """
        return list(NAME_PARTITION_TO_RECORID.keys())
        
    @classmethod
    def getLocallyAvailableEntries(cls, benchmarkDir: Union[str, PathLike] = defaultBenchmarkDir) -> List[Tuple[str, int]]:
        """
        Returns the list of available entries in benchmarkDir
        Args:
            benchmarkDir (str): The root directory where the datasets are downloaded.

        Returns:
            List[Tuple[str,int]]: the list of available entries in benchmarkDir

        """
        avail = []
        for dirname in os.listdir(benchmarkDir):
            if os.path.isfile(dirname):
                continue
            if os.path.exists(os.path.join(benchmarkDir, dirname, "particles_0.mrcs")):
                avail.append((dirname, 0))
            if os.path.exists(os.path.join(benchmarkDir, dirname, "particles_1.mrcs")):
                avail.append((dirname, 1))
        return avail

    def _download(self):
        assert (self.targetName, self.halfset) in NAME_PARTITION_TO_RECORID, (f"Error, unknown target and/or "
                                                                              f"halfset {self.targetName} {self.halfset}")
        download_record(NAME_PARTITION_TO_RECORID[self.targetName, self.halfset],
                        destination_dir=self.datadir)
        #Validation
        pset = ParticlesStarSet(starFname=self.starFname, particlesDir=self.datadir)
        for i in tqdm(range(len(pset)), desc="Checking downloaded dataset"):
            img, md = pset[i]
            assert len(img.shape) == 2, f"Error, there were problems downloading the target {self.targetName}"

    def _is_avaible(self):
        return osp.isfile(getDoneFname(self.datadir,
                                       NAME_PARTITION_TO_RECORID.get((self.targetName, self.halfset), self.halfset)))


def _merge_star_files(star_fnames: List[Union[str, PathLike]], merged_fname: Union[str, PathLike]):
    """
    Merge multiple particles star files that may differ in headers.
    - Optics tables are concatenated and deduplicated (union of columns).
    - Particle tables are concatenated using the union of columns.
    """
    assert len(star_fnames) >= 2, "Error, at least two star files are needed to merge"

    def _split_blocks(star_data):
        if isinstance(star_data, dict):
            optics_key = next((k for k in star_data if "optic" in k.lower()), None)
            particle_key = next((k for k in star_data if "particle" in k.lower() and k != optics_key), None)
            if particle_key is None:
                particle_key = next(iter({k for k in star_data if k != optics_key}), None)
            optics_df = star_data.get(optics_key) if optics_key else None
            particles_df = star_data.get(particle_key) if particle_key else None
        else:
            optics_df = None
            particles_df = star_data
        assert particles_df is not None, "Error, could not identify particle table in star file"
        return optics_df, particles_df

    optics_list = []
    particles_list = []
    for fname in star_fnames:
        optics_df, particles_df = _split_blocks(starfile.read(fname))
        optics_list.append(optics_df)
        particles_list.append(particles_df)

    all_particle_cols = sorted(set().union(*(set(df.columns) for df in particles_list)))
    particles_list = [df.reindex(columns=all_particle_cols) for df in particles_list]
    merged_particles = pd.concat(particles_list, ignore_index=True)

    merged_data = {"particles": merged_particles}

    has_optics = any(df is not None for df in optics_list)
    if has_optics:
        optics_list = [df if df is not None else pd.DataFrame() for df in optics_list]
        all_optics_cols = sorted(set().union(*(set(df.columns) for df in optics_list)))
        optics_list = [df.reindex(columns=all_optics_cols) for df in optics_list]
        merged_optics = pd.concat(optics_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
        merged_data["optics"] = merged_optics

    starfile.write(merged_data, merged_fname, overwrite=True)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(osp.join(default_configs_dir, "defaultDataConfig.yaml"))

    parser = ArgumentParser(description="Dataset utility")
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)

    # Create parser for 'visualize' mode
    visualize_parser = subparsers.add_parser("visualize", help="Run dataset visualization")

    visualize_parser.add_argument("-t", "--targetName", help="The target to use", type=str, required=True)
    visualize_parser.add_argument("-p", "--halfset", help="The halfset to use", choices=["0", "1"], default="0")
    visualize_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str,
                                  default=defaultBenchmarkDir)
    visualize_parser.add_argument("-s", "--image_size", help="The desired image size", type=int,
                                  default=cfg.data.image_size)
    visualize_parser.add_argument("-c", "--image_size_factor_for_crop", help="Percentage of image to crop", type=float,
                                  default=cfg.data.image_size_factor_for_crop)
    visualize_parser.add_argument("-f", "--phase_flippling", help="Apply phase flippling", action="store_true")
    visualize_parser.add_argument("-i", "--channels_to_show", help="Channels of the images to show", type=int,
                                  nargs="+")

    # Create parser for 'add_entry' mode
    add_entry_parser = subparsers.add_parser("add_entry", help="Run add new entry locally")
    add_entry_parser.add_argument("--newTargetName", help="The name of the new target to use", type=str,
                                  required=True)
    add_entry_parser.add_argument("-p", "--halfset", help="The halfset to use", choices=["0", "1"],
                                  required=True)
    add_entry_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str,
                                  default=defaultBenchmarkDir)
                                  
    add_entry_parser.add_argument("--starFname",
                                  help="The star filename with the particles to be added to the local benchmark", type=str, required=True)
    add_entry_parser.add_argument("--particlesRootDir", help="The root directory referred to in the starFname", type=str, required=True)
    add_entry_parser.add_argument("--symmetry", help="The point symmetry of the dataset", type=str, required=True)



    list_entries_parser = subparsers.add_parser("list_entries", help="List available entries")
    list_entries_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str, default=defaultBenchmarkDir)


    donwload_entry_parser = subparsers.add_parser("download_entry", help="Download an entry")
    donwload_entry_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str, default=defaultBenchmarkDir)
    donwload_entry_parser.add_argument("-p", "--halfset", help="Halfset to download (omit to fetch both)", choices=["0", "1"], required=False, default=None)
    donwload_entry_parser.add_argument("-t", "--targetName", help="The target to use", type=str, required=True)
    
    preprocess_entry_parser = subparsers.add_parser("preprocess_entry", help="Preprocess an entry")
    preprocess_entry_parser.add_argument("-b", "--benchmarkDir", help="The benchmark's directory", type=str, default=defaultBenchmarkDir)
    preprocess_entry_parser.add_argument("-p", "--halfset", help="The halfset to use", choices=["0", "1"], required=True)
    preprocess_entry_parser.add_argument("-t", "--targetName", help="The target to use", type=str, required=True)
    preprocess_entry_parser.add_argument("-o", "--outDir", help="The output directory", type=str, required=True)
    
    preprocess_entry_parser.add_argument("-s", "--image_size", help="The final size of the image in pixels, obtained by resizing. Default: Original image size", type=int, required=False, default=None)
    preprocess_entry_parser.add_argument("-f", "--ctf_correction", help="The ctf correction mode. Default: %(default)s", choices=['none', 'phase_flip'], default="none")
    preprocess_entry_parser.add_argument("-c", "--image_size_factor_for_crop", help="The fraction of the original image to be cropped. Default: ", type=float, required=False, default=0.2)

    args = parser.parse_args()

    if args.mode == "visualize":
        # Run dataset visualization
        ps = ParticlesDataset(targetName=args.targetName,
                              halfset=int(args.halfset),
                              benchmarkDir=args.benchmarkDir,
                              image_size=args.image_size,
                              ctf_correction="phase_flip" if args.phase_flippling else "none",
                              image_size_factor_for_crop=args.image_size_factor_for_crop
                              )
        import matplotlib.pyplot as plt

        channels_to_show = args.channels_to_show if args.channels_to_show else [0]
        for elem in ps:
            iid, img, *_ = elem
            assert 1 <= len(channels_to_show) <= 4, "Error, at least one channel required and no more than 4"
            f, axes = plt.subplots(1, len(channels_to_show), squeeze=False)
            for j, c in enumerate(channels_to_show):
                axes[0, c].imshow(img[c, ...], cmap="gray")
            plt.show()
            plt.close()
        print("Done")

    elif args.mode == "add_entry":
        # Run addNewEntryLocally
        ParticlesDataset.addNewEntryLocally(starFname=args.starFname,
                                            particlesRootDir=args.particlesRootDir,
                                            newTargetName=args.newTargetName,
                                            halfset=int(args.halfset),
                                            symmetry=args.symmetry,
                                            benchmarkDir=args.benchmarkDir)
        print(
            f"Successfully added new entry {args.newTargetName} with halfset {args.halfset} to benchmark "
            f"directory {args.benchmarkDir}.")

    elif args.mode == "list_entries":
        remote_entries = ParticlesDataset.getCESPEDEntries()
        print("Available for donwload entries:", remote_entries)
        local_entries = ParticlesDataset.getLocallyAvailableEntries(benchmarkDir=args.benchmarkDir)
        print("Locally available entries:", local_entries)
        
    elif args.mode == "download_entry":

        halfsets = [int(args.halfset)] if args.halfset is not None else [0, 1]
        downloaded = []
        for hs in halfsets:
            ps = ParticlesDataset(targetName=args.targetName,
                                  halfset=hs,
                                  benchmarkDir=args.benchmarkDir,
                                  image_size=None,
                                  ctf_correction="none",
                                  image_size_factor_for_crop=0.,
                                  )
            # Trigger download/validation
            ps[0]
            downloaded.append(ps)
            print(f"Downloaded halfset {hs} to:")
            print(ps.starFname)
            print(ps.stackFname)

        if len(halfsets) > 1:
            datadir = downloaded[0].datadir
            merged_star = osp.join(datadir, "particles_merged.star")
            star_paths = [osp.join(datadir, f"particles_{hs}.star") for hs in halfsets]
            _merge_star_files(star_paths, merged_star)
            print("Merged star file created at:")
            print(merged_star)
    
    elif args.mode == "preprocess_entry":
        ps = ParticlesDataset(targetName=args.targetName,
                              halfset=int(args.halfset),
                              benchmarkDir=args.benchmarkDir,
                              image_size=args.image_size,
                              ctf_correction=args.ctf_correction,
                              image_size_factor_for_crop=args.image_size_factor_for_crop
                              )
                              
        stack = ParticlesStarSet(starFname=ps.starFname)
        assert isinstance(stack[len(stack) - 1][0], np.ndarray), "Error, there is some problem reading your data"
        
        outDir = os.path.expanduser(args.outDir)
        os.makedirs(outDir, exist_ok=True)
        newStarFname = os.path.join(outDir, f"particles_{args.halfset}.star")
        
        stack.createFromPdNp(newStarFname, stack.optics_md,  stack.particles_md, npImages=(row[1] for row in ps), overwrite=True)

    
    else:
        raise ValueError("Error, option is not valid")

    """
    Reunning example
    
    particlesDataset visualize --targetName TEST --halfset 0
    """
