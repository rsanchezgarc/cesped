# CESPED: Utilities for the Cryo-EM Supervised Pose Estimation Dataset

#TODO: Write this readme.
#TODO: Refactor


## Installation
cesped has been tested on python 3.11. Installation should be automatic using the requirements.txt file
```
cd cesped
pip install .
```


## Basic usage

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
For a rapid test, use `targetName="TEST` and `halfset=0`

3. Use it as a regular dataset
```
dl = Dataloader(datatset, ds)
for batch in dl:
  iid, img, (rotMat, xyShiftAngs, confidence), metadata = batch
  
  #iid is the id of the particle (an string)
  #img is a batch of Bx1xNxN images
  #rotMat is a batch of rotation matrices Bx3x3
  #xyShiftAngs is a batch of image shifts in Angstroms Bx2
  #confidence is a batch of numbers, between 0 and 1, Bx1
  #metata is a dictionary of names:values for all the information about the particle
  
  #YOUR PYTORCH CODE HERE
  
```

## Image2Sphere experiments
The experiments have been implemented using lightning and lightingCLI. You can find the configuration files 
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
python -m cesped.trainEntry --data.halfset <HALFSET> --data.targetName <TARGETNAME>
```
with `<HALFSET>` 0 or 1 and `<TARGETNAME>` one of the list that can be found using `ParticlesDataset.getCESPEDEntries()`
Some available targets include
- TEST. A small subset of EMPIAR-10166
- 10166. The EMPIAR-10166
- 11120. The EMPIAR-11120

Do not forget to change the configuration files or to provide different values via command line or environmental 
variables. In addition, `[--config CONFIG_NAME.yaml]` also allows to overwrite the default values using (a/some) custom
yaml file(s). Use `-h` to see the list of configurable parameters. Some of the most important ones are.
- trainer.default_root_dir. Directory where the checkpoints and the logs will be saved, 
from [defaultTrainerConfig.yaml](cesped%2Fconfigs%2FdefaultTrainerConfig.yaml)
- optimizer.lr. The learning rate, from [defaultOptimizerConfig.yaml](cesped%2Fconfigs%2FdefaultOptimizerConfig.yaml)
- data.benchmarkDir. Directory where the benchmark entries are saved, from [defaultModelConfig.yaml](cesped%2Fconfigs%2FdefaultModelConfig.yaml)
- data.num_data_workers. Number of workers for data loading, from [defaultModelConfig.yaml](cesped%2Fconfigs%2FdefaultModelConfig.yaml)

### Inference
In order to predict the poses on one target, you run
```
python -m cesped.inferEntry --data.halfset <HALFSET> --data.targetName <TARGETNAME> --ckpt_path <PATH_TO_CHECKPOINT>
```

##API
For API documentation run
pdoc --http : .

