# Equibind score

Equibind score extracts the final hidden layer from the ligand graph in equibind, and then passes this through a GNN to predict binding affinity (instead of docked pose as in the original equibind).

First run 'get_layer.py' with input argument as the name of the binary file you want the hidden layer graphs to be stored in (.bin extension recommended). First make sure you edit config.yml to point towards the correct folder.

Next run inference.py with this binary file as an argument and it will output the predicted binding affinities as a text file.