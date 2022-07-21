import models.score_model as score_model
from commons.custom_data_loader import custom_loader, custom_collate_10, custom_collate_20, custom_collate_11, custom_collate_21
import pandas as pd
from copy import deepcopy
import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# Turn off autograd
torch.set_grad_enabled(False)

# Parse arguments
def parse_arguments(arglist = None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yml')
    p.add_argument('--hidden_layers', type=str, default='hidden_layers/test', help='path to hidden binary layers')
    p.add_argument('--type', type=str, default='both', help='ligand, receptor, or both: which graph/model will you use (make sue this aligns with the model you load)')
    p.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    p.add_argument('--model_input', type=str, default='runs/score/ligand_trained.pt', help='path to model you want to run')
    p.add_argument('--output_file', type=str, default='inference.csv', help='path to csv file where binding affinities will be saved')

    cmdline_parser = deepcopy(p)
    args = p.parse_args(arglist)
    clear_defaults = {key: argparse.SUPPRESS for key in args.__dict__}
    cmdline_parser.set_defaults(**clear_defaults)
    cmdline_parser._defaults = {}
    cmdline_args = cmdline_parser.parse_args(arglist)
        
    return args, cmdline_args

args, cmdline_args = parse_arguments()
if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            if key in cmdline_args:
                continue
            arg_dict[key] = value
    args.config = args.config.name

# model 1 - true, model 2 - false
if args.type == 'ligand' or args.type == 'receptor':
    model1 = True
else:
    model1 = False

# Set gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
if model1:
    model = score_model.GAT1()
else:
    model = score_model.GAT2()
model.load_state_dict(torch.load(args.model_input))
model.to(device)
model.eval()

# Load the data
data = custom_loader(args.hidden_layers,args.type)
if model1:
    loader = DataLoader(data,shuffle=False,batch_size=args.batch_size,collate_fn=custom_collate_10)
else:
    loader = DataLoader(data,shuffle=False,batch_size=args.batch_size,collate_fn=custom_collate_20)

# Initialise dataframe to save results
df = pd.DataFrame(columns=['pk'])

# Perform the inference (batched because sometimes the graphs don't fit in memory)
if model1:
    for batchedgraph, namels in loader:

        pred = model(batchedgraph,batchedgraph)
        df.append(pd.DataFrame(pred, columns=['pk'], index=namels))
else:
    for ligbatchedgraph, recbatchedgraph, namels in loader:

        pred = model(ligbatchedgraph,recbatchedgraph)
        df = df.append(pd.DataFrame(pred, columns=['pk'], index=namels))

# Write predictions to file
csv = 'PDB'
csv += df.to_csv()
file = open(args.output_file, 'w')
file.write(csv)
file.close()