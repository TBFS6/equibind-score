# Run this script first to create the .bin files containing the hidden layer graphs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:28:47 2022

@author: rory
"""

import argparse
import sys

from copy import deepcopy

import os

from dgl import load_graphs

from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm

from commons.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, apply_changes, random_rotation_translation
from commons.logger import Logger
from commons.process_mols import read_molecule, get_lig_graph_revised, \
    get_rec_graph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference

from datasets.pdbbind import PDBBind

from commons.utils import seed_all, read_strings_from_txt

import yaml

from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
#from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader
from dgl.data.utils import save_graphs


def load_model(args, data_sample, device, **kwargs):
    model = globals()[args.model_type](device=device,
                                           lig_input_edge_feats_dim=data_sample[0].edata['feat'].shape[1],
                                           rec_input_edge_feats_dim=data_sample[1].edata['feat'].shape[1],
                                           **args.model_parameters, **kwargs)
    return model

def hiddenlayer():

    # The following is all just setup stuff can probably be left alone
    ###
    def parse_arguments(arglist = None):
        p = argparse.ArgumentParser()

        p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config.yml')

        p.add_argument('--checkpoint', type=str, help='path to .pt file in a checkpoint directory')
        p.add_argument('--output_directory', type=str, default=None, help='path where to put the predicted results')
        p.add_argument('--run_corrections', type=bool, default=False,
                    help='whether or not to run the fast point cloud ligand fitting')
        p.add_argument('--run_dirs', type=list, default=[], help='path directory with saved runs')
        p.add_argument('--fine_tune_dirs', type=list, default=[], help='path directory with saved finetuning runs')
        p.add_argument('--inference_path', type=str, help='path to some pdb files for which you want to run inference')
        p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
        p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
        p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
        p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
        p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
        p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
        p.add_argument('--dataset_params', type=dict, default={},
                    help='parameters with keywords of the dataset')
        p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
        p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
        p.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
        p.add_argument('--multithreaded_seeds', type=list, default=[],
                    help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
        p.add_argument('--seed_data', type=int, default=1, help='if you want to use a different seed for the datasplit')
        p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
        p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
        p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
        p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
        p.add_argument('--clip_grad', type=float, default=None, help='clip gradients if magnitude is greater')
        p.add_argument('--lr_scheduler', type=str,
                    help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
        p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
        p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                    help='step every batch if true step every epoch otherwise')
        p.add_argument('--log_iterations', type=int, default=-1,
                    help='log every log_iterations iterations (-1 for only logging after each epoch)')
        p.add_argument('--expensive_log_iterations', type=int, default=100,
                    help='frequency with which to do expensive logging operations')
        p.add_argument('--eval_per_epochs', type=int, default=0,
                    help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
        p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
        p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
        p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
        p.add_argument('--val_per_batch', type=bool, default=True,
                    help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
        p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
        p.add_argument('--num_epochs_local_only', type=int, default=1,
                    help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

        p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
        p.add_argument('--collate_params', type=dict, default={},
                    help='parameters with keywords of the chosen collate function')
        p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

        p.add_argument('--models_to_save', type=list, default=[],
                    help='specify after which epochs to remember the best model')

        p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
        p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

        p.add_argument('--trainer', type=str, default='binding', help='')
        p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

        p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
        p.add_argument('--check_se3_invariance', type=bool, default=False, help='check it instead of generating files')
        p.add_argument('--num_confs', type=int, default=1, help='num_confs if using rdkit conformers')
        p.add_argument('--use_rdkit_coords', action="store_true",
                    help='override the rkdit usage behavior of the used model')
        p.add_argument('--no_use_rdkit_coords', action="store_false", dest = "use_rdkit_coords",
                    help='override the rkdit usage behavior of the used model')

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

    for run_dir in args.run_dirs:
        args.checkpoint = f'runs/{run_dir}/best_checkpoint.pt'
        config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
        # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if (key not in config_dict.keys()) and (key not in cmdline_args):
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        args.model_parameters['noise_initial'] = 0
    ###

    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    checkpoint = torch.load('runs/flexible_self_docking/best_checkpoint.pt', map_location=device)
    model = None
    all_ligs_coords_corrected = []
    all_intersection_losses = []
    all_intersection_losses_untuned = []
    all_ligs_coords_pred_untuned = []
    all_ligs_coords = []
    all_ligs_keypts = []
    all_recs_keypts = []
    all_names = []
    dp = args.dataset_params
    use_rdkit_coords = args.use_rdkit_coords if args.use_rdkit_coords != None else args.dataset_params[
        'use_rdkit_coords']
    names = os.listdir(args.inference_path) if args.inference_path != None else tqdm(read_strings_from_txt('data/timesplit_test'))

    # rory's code
    ###
    graphls = []
    graph_labels = {}
    ###

    for idx, name in enumerate(names):
        print(f'\nProcessing {name}: complex {idx + 1} of {len(names)}')
        file_names = os.listdir(os.path.join(args.inference_path, name))
        rec_name = [i for i in file_names if 'rec.pdb' in i or 'protein' in i][0]
        lig_names = [i for i in file_names if 'ligand' in i]
        rec_path = os.path.join(args.inference_path, name, rec_name)
        for lig_name in lig_names:
            if not os.path.exists(os.path.join(args.inference_path, name, lig_name)):
                raise ValueError(f'Path does not exist: {os.path.join(args.inference_path, name, lig_name)}')
            print(f'Trying to load {os.path.join(args.inference_path, name, lig_name)}')
            lig = read_molecule(os.path.join(args.inference_path, name, lig_name), sanitize=True)
            if lig != None:  # read mol2 file if sdf file cannot be sanitized
                used_lig = os.path.join(args.inference_path, name, lig_name)
                break
        if lig_names == []: raise ValueError(f'No ligand files found. The ligand file has to contain \'ligand\'.')
        if lig == None: raise ValueError(f'None of the ligand files could be read: {lig_names}')
        #print(f'Docking the receptor {os.path.join(args.inference_path, name, rec_name)}\nTo the ligand {used_lig}')

        rec, rec_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(rec_path)
        rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                use_rec_atoms=dp['use_rec_atoms'], rec_radius=dp['rec_graph_radius'],
                                surface_max_neighbors=dp['surface_max_neighbors'],
                                surface_graph_cutoff=dp['surface_graph_cutoff'],
                                surface_mesh_cutoff=dp['surface_mesh_cutoff'],
                                c_alpha_max_neighbors=dp['c_alpha_max_neighbors'])
        lig_graph = get_lig_graph_revised(lig, name, max_neighbors=dp['lig_max_neighbors'],
                                        use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
        if 'geometry_regularization' in dp and dp['geometry_regularization']:
            geometry_graph = get_geometry_graph(lig)
        elif 'geometry_regularization_ring' in dp and dp['geometry_regularization_ring']:
            geometry_graph = get_geometry_graph_ring(lig)
        else:
            geometry_graph = None

        # move graphs to cuda
        rec_graph.to(device)
        lig_graph.to(device)

        start_lig_coords = lig_graph.ndata['x']
        # Randomly rotate and translate the ligand.
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        if (use_rdkit_coords):
            lig_coords_to_move = lig_graph.ndata['new_x']
        else:
            lig_coords_to_move = lig_graph.ndata['x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        input_coords = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        lig_graph.ndata['new_x'] = input_coords

        if model == None:
            model = load_model(args, data_sample=(lig_graph, rec_graph), device=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

        # rory's code
        print('Calculating equibind layer')
        layer = model.forward(lig_graph,rec_graph,geometry_graph)
        lig_graph.ndata['final_hidden'] = layer
        lig_graph.requires_grad = False

        del lig_graph.ndata['x']
        del lig_graph.ndata['feat']
        del lig_graph.ndata['mu_r_norm']
        del lig_graph.ndata['new_x']
        del lig_graph.ndata['x_now']
        del lig_graph.ndata['x_update']
        del lig_graph.ndata['aggr_msg']
        del lig_graph.edata['feat']
        del lig_graph.edata['msg']
        del lig_graph.edata['x_rel']

        graphls.append(lig_graph)

        graph_labels[name] = torch.tensor([idx])

    return graphls, args, graph_labels
    

if __name__ == '__main__':
    graphls, args, graph_labels = hiddenlayer()
    outname = args.output_directory
    save_graphs(outname,graphls,graph_labels)