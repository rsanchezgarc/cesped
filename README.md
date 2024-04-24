# CESPED: Utilities for the Cryo-EM Supervised Pose Estimation Dataset

CESPED is a new dataset specifically designed for Supervised Pose Estimation in Cryo-EM. You can check our manuscript at https://arxiv.org/abs/2311.06194.

## Installation
cesped has been tested on python 3.11. Installation should be automatic using pip
```
pip install cesped
#Or directy from the master branch
pip install git+https://github.com/rsanchezgarc/cesped
```

or cloning the repository
```
git clone https://github.com/rsanchezgarc/cesped
cd cesped
pip install .
```


## Basic usage

### ParticlesDataset class
It is used to load the images and poses.

1. Get the list of downloadable entries
```
from cesped.particlesDataset import ParticlesDataset
listOfEntries = ParticlesDataset.getCESPEDEntries()
```
2. Load a given entry
```
targetName, halfset = listOfEntries[0] #We will work with the first entry only

dataset = ParticlesDataset(targetName, halfset)
```
For a rapid test, use `targetName="TEST"` and `halfset=0`. If the dataset is not yet available in the benchmarkDir (defined in [defaultDataConfig.yaml](cesped%2Fconfigs%2FdefaultDataConfig.yaml),
it will be automatically downloaded. Metadata (Euler angles, CTF,...) are stored using Relion starfile format, and images are stored as .mrcs stacks.

3. Use it as a regular dataset
```
dl = DataLoader(dataset, batch_size=32)
for batch in dl:
  iid, img, (rotMat, xyShiftAngs, confidence), metadata = batch
  
  #iid is the list of ids of the particles (string)
  #img is a batch of Bx1xNxN images
  #rotMat is a batch of rotation matrices Bx3x3
  #xyShiftAngs is a batch of image shifts in Angstroms Bx2
  #confidence is a batch of numbers, between 0 and 1, Bx1
  #metata is a dictionary of names:values for all the information about the particle
  
  #YOUR PYTORCH CODE HERE
  predRot = model(img)
  loss = loss_function(predRot, rotMat)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  
```

4. Once your model is trained, you can update the metadata of the ParticlesDataset and save it so that it can be used in cryo-EM software
```
for iid, pred_rotmats, maxprob in predictions:
    #iid is the list of ids of the particles (string)
    #pred_rotmats is a batch of predicted rotation matrices Bx3x3
    #maxprob is a batch of numbers, between 0 and 1, Bx1, that indicates the confidence in the prediction (e.g. softmax values)

    particlesDataset.updateMd(ids=iid, angles=pred_rotmats,
                              shifts=torch.zeros(pred_rotmats.shape[0],2, device=pred_rotmats.device), #Or actual predictions if you have them
                              confidence=maxprob,
                              angles_format="rotmat")
particlesDataset.saveMd(outFname) #Save the metadata as an starfile, a common cryo-EM format

  
```
5. Finally, evaluation can be computed if the predictions for the halfset 0 and halfset 1 were saved using the evaluateEntry script.
```
python -m cesped.evaluateEntry  --predictionType SO3 --targetName 11120  \
--half0PredsFname particles_preds_0.star  --half1PredsFname particles_preds_1.star \
--n_cpus 12 --outdir evaluation/
```
evaluateEntry uses [Relion](https://relion.readthedocs.io/) for reconstruction, so you will need to install it and 
edit the config file [defaultRelionConfig.yaml](cesped%2Fconfigs%2FdefaultRelionConfig.yaml) or provide, via command 
line arguments, where Relion is installed
```
--mpirun /path/to/mpirun  --relionBinDir /path/to/relion/bin
```
Alternatively, you can build a [singularity](https://docs.sylabs.io/guides/3.0/user-guide/index.html) image, using the
definition file we provide [relionSingularity.def](cesped%2FrelionSingularity.def)
```commandline
singularity build relionSingularity.sif relionSingularity.def
```
and edit the config file to point where the singularity image file is located, or use the command line argument
```
--singularityImgFile /path/to/relionSingularity.sif
```

### Cross-plataform usage.

Users of other deep learning frameworks can download CESPED entries using the following command

```
python -m cesped.particlesDataset download_entry -t 10166 --halfset 0
```
This will download the associated starfile and mrcs file to the default benchmark directory (defined in [defaultDataConfig.yaml](cesped%2Fconfigs%2FdefaultDataConfig.yaml).
Use `--benchmarkDir` to specify another directory<br/>

In order to list the entries available for download and the ones already downloaded, you can use
```
python -m cesped.particlesDataset list_entries
```
Preprocessing of the dataset entries can be executed using
```
python -m cesped.particlesDataset preprocess_entry --t 10166 --halfset 0 --o /tmp/dumpedData/ --ctf_correction "phase_flip"
```
where `--t` is the target name. Use `-h` to display the list of available preprocessing operations.

The raw data can be easily accessed using the Python package [starstack](https://pypi.org/project/starstack/), which relies on the [mrcfile](https://pypi.org/project/mrcfile/) and [starfile](https://pypi.org/project/starfile/) packages. Predictions should be written as a star file with the newly
predicted Euler angles.

Evaluation can be computed once the predictions for the half-set 0 and half-set 1 are saved

```
python -m cesped.evaluateEntry  --predictionType SO3 --targetName 11120  \
--half0PredsFname particles_preds_0.star  --half1PredsFname particles_preds_1.star \
--n_cpus 12 --outdir evaluation/
```

## Image2Sphere experiments
The experiments have been implemented using [lightning](https://lightning.ai/) and lightingCLI. You can find the configuration files 
located at :
```
YOUR_DIR/cesped/configs/
```
You can also find it as:
```
import cesped
cesped.default_configs_dir
```
### Train
In order to train the model on one target, you run
```
python -m cesped.trainEntry --data.halfset <HALFSET> --data.targetName <TARGETNAME> --trainer.default_root_dir <OUTDIR>
```
with `<HALFSET>` 0 or 1 and `<TARGETNAME>` one of the list that can be found using `ParticlesDataset.getCESPEDEntries()` 
<br><br>
The included targets are:


| EMPIAR ID | Composition | Symmetry | Image Pixels | FSCR<sub>0.143</sub> (Å) | Masked FSCR<sub>0.143</sub> (Å) | # Particles |
|-----------|-------------|----------|--------------|-------------------------|---------------------------------|-------------|
| 10166     | Human 26S proteasome bound to the chemotherapeutic Oprozomib | C1 | 284 | 5.0 | 3.9 | 238631 |
| 10786     | Substance P-Neurokinin Receptor G protein complexes (SP-NK1R-miniGs399) | C1 | 184 | 3.3 | 3.0* | 288659 |
| 10280     | Calcium-bound TMEM16F in nanodisc with supplement of PIP2 | C2 | 182 | 3.6 | 3.0* | 459504 |
| 11120     | M22 bound TSHR Gs 7TM G protein | C1 | 232 | 3.4 | 3.0* | 244973 |
| 10648     | PKM2 in complex with Compound 5 | D2 | 222 | 3.7 | 3.3 | 234956 |
| 10409     | Replicating SARS-CoV-2 polymerase (Map 1) | C1 | 240 | 3.3 | 3.0* | 406001 |
| 10374     | Human ABCG2 transporter with inhibitor MZ29 and 5D3-Fab | C2 | 216 | 3.7 | 3.0* | 323681 |

`*` Nyquist Frequency at 1.5 Å/pixel; Resolution is estimated at the usual threshold 0.143.  
Reported FSCR<sub>0.143</sub> values were obtained directly from the relion_refine logs while Masked FSCR<sub>0.143</sub> values were collected from the relion_postprocess logs.

In addition, the entry TEST is a small subset of EMPIAR-11120

Do not forget to change the configuration files or to provide different values via the command line or environmental 
variables. In addition, `[--config CONFIG_NAME.yaml]` also allows overwriting the default values using (a/several) custom
yaml file(s). Use `-h` to see the list of configurable parameters. Some of the most important ones are.
- trainer.default_root_dir. Directory where the checkpoints and the logs will be saved, 
from [defaultTrainerConfig.yaml](cesped%2Fconfigs%2FdefaultTrainerConfig.yaml)
- optimizer.lr. The learning rate, from [defaultOptimizerConfig.yaml](cesped%2Fconfigs%2FdefaultOptimizerConfig.yaml)
- data.benchmarkDir. Directory where the benchmark entries are saved, from [defaultDataConfig.yaml](cesped%2Fconfigs%2FdefaultDataConfig.yaml). It is recommended
to change this in the config file.
- data.num_data_workers. Number of workers for data loading, from [defaultDataConfig.yaml](cesped%2Fconfigs%2FdefaultDataConfig.yaml)
- data.batch_size. from [defaultDataConfig.yaml](cesped%2Fconfigs%2FdefaultDataConfig.yaml)

### Inference
By default, when using `python -m cesped.trainEntry`, inference on the complementary halfset is done on a single GPU
after training finishes, and the starfile with the predictions can be found at 
`<OUTDIR>/lightning_logs/version_<\d>/predictions_[0,1].star`. In order to manually run the pose prediction 
code (and to make use of all GPUs) you can run
```
python -m cesped.inferEntry --data.halfset <HALFSET> --data.targetName <TARGETNAME> --ckpt_path <PATH_TO_CHECKPOINT> \
--outFname /path/to/output/starfile.star
```
### Evaluation
5. As before, evaluation can be computed if the predictions for the halfset 0 and halfset 1 were saved using the evaluateEntry script.
```
python -m cesped.evaluateEntry  --predictionType SO3 --targetName 11120  \
--half0PredsFname particles_preds_0.star  --half1PredsFname particles_preds_1.star \
--n_cpus 12 --outdir evaluation/
```

## API

For API documentation check the [docs folder](https://rsanchezgarc.github.io/cesped/cesped/)


## Relion Singularity

A singularity container for relion_reconstruct with MPI support can be built with the following command. 
```
singularity build relionSingulary.sif relionSingulary.def 
```
Then, Relion reconstruction can be computed with the following command:
```
singularity exec relionSingulary.sif mpirun -np 4 relion_reconstruct_mpi --ctf --pad 2 --i input_particles.star --o output_map.mrc
#Or the following command
./relionSingulary.sif  4 --ctf --pad 2 --i input_particles.star --o output_map.mrc #This uses 4 mpis
```
However, typical users will not need to execute the container manually. Everything happens transparently within the evaluateEntry.py script
