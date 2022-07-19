# Equibind score

Equibind score can extract different feature graphs from equibind, then can pass them through a GNN to predict binding affinity (instead of docked pose as in the original equibind).

### Get hidden layers
First run 'get_layer.py' to create the hidden layer binary files. First make sure you edit config.yml to point towards the correct input and output folder, or specify the options via the command line. You can specify whether you want to extract the ligand graph, receptor graph, or both.

### Inference
Not quite finished

### Training
Finshed