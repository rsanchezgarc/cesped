# CESPED: Utilities for the Cryo-EM Supervised Pose Estimation Dataset

#TODO: Write this readme.
#TODO: Finish data augmentation
#TODO: Start uploading full targegs
#TODO: Refactor


## Installation
```
cd cesped
pip install .
```


## Basic usage

1. Get the list of downloadable entries
```
listOfEntries = ParticlesDataset.getCESPEDEntries()
```
2. Load a given entry
```
targetName, halfset = listOfEntries[0] #We will work with the first example

dataset = ParticlesDataset(targetName, halfset)
```

3. Use is a regular dataset
```
dl = Dataloader(datatset, ds)
for batch in dl:
  iid, img, (rotMat, xyShiftAngs, confidence), metadata = batch
  #YOUR PYTORCH CODE HERE
```

For documentation run
pdoc --http : .

