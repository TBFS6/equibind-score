# Equibind score

Equibind score can extract different feature graphs from equibind, then can pass them through a GNN to predict binding affinity (instead of docked pose as in the original equibind).

### Setup Environment

We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). Clone the
current repo

    git clone https://github.com/HannesStark/EquiBind

Create a new environment with all required packages using 'environment.yml'. If you have a CUDA GPU run:

    conda env create -f environment.yml

If you instead only have a CPU run:

    conda env create -f environment_cpuonly.yml

Activate the environment

    conda activate equibind

Here are the requirements themselves for the case with a CUDA GPU if you want to install them manually instead of using the 'environment.yml':
````
python=3.7
pytorch 1.10
torchvision
cudatoolkit=10.2
torchaudio
dgl-cuda10.2
rdkit
openbabel
biopython
rdkit
biopandas
pot
dgllife
joblib
pyaml
icecream
matplotlib
tensorboard
````

### Get hidden layers
Make sure you edit 'configs/get_layer.yml' to point towards the correct input and output folder, or specify the options via the command line. You can specify whether you want to extract the ligand graph, receptor graph, or both.

Then run:

    python get_layer.py

### Inference
In this repository there are 3 pretrained models: one which runs on the ligand feature graph from EquiBind, one on the receptor feature graph, and one which runs on both graphs. I'm still doing some hyperparameter tuning so these will hopefully improve but currently the model which runs on both graphs works best and has a validation RMSE of around 1.65pK.

Make sure you edit 'configs/inference.yml' to point towards the correct pretrained model, the hidden layer folder, the output file where the predictions will be saved in a .csv file, and the batch size for inference. Note the batch size is used because on large input folders you may run out of RAM if the entire inference is done batched. The hidden layer folder should contain two folders: 'ligand' and 'receptor', each containing the graph files named as follows: '*PDB*.bin', replacing *PDB* with the correct PDBBind code.

To perform inference just run:

    python inference.py

### Training
First edit 'configs/trainer.yml'. Note your hidden layers folder must have three folders in it: 'train', 'val' and 'test', each containing a 'ligand' folder and a 'receptor' folder as above. I may implement automatic test/train splits for other datasets but wanted to train it on the same split as EquiBind with PDBBind.

To train just run:

    python trainer.py