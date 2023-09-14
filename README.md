# CESPED: Utilities for the Cryo-EM Supervised Pose Estimation Dataset

#TODO: Write this readme.


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



For documentation run
pdoc --http : .

