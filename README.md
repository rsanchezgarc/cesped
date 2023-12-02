# CESPED: Utilities for the Cryo-EM Supervised Pose Estimation Dataset

CESPED, is a new dataset specifically designed for Supervised Pose Estimation in Cryo-EM. You can check of manuscript at https://arxiv.org/abs/2311.06194.

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
For a rapid test, use `targetName="TEST"` and `halfset=0`

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
Some available targets include
- TEST. A small subset of EMPIAR-10166
- 10166. The EMPIAR-10166
- 11120. The EMPIAR-11120
- 10280. The EMPIAR-10280
- 10409. The EMPIAR-10409

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

##API
For API documentation run
pdoc --http : .




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
