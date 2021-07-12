from __future__ import print_function, absolute_import
import os
import argparse
import random
import numpy as np
# torch-related packages
import torch
# import matplotlib.pyplot as plt
# from utils.visualization import visualize_TSNE
torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# data
from data_loader import Visda_Dataset, Office_Dataset, Home_Dataset, Visda18_Dataset
from model_trainer_new import ModelTrainer
from src_model_trainer_new import SRCModelTrainer
from utils.logger import Logger
from utils.visualization import draw_reliability_graph

def set_exp_name(args):
    exp_name = 'D-{}'.format(args.dataset)
    if args.dataset == 'office' or args.dataset == 'home':
        exp_name += '_src-{}_tar-{}'.format(args.source_name, args.target_name)
    exp_name += '_A-{}'.format(args.arch)
    exp_name += '_L-{}'.format(args.num_layers)
    exp_name += '_E-{}_B-{}'.format(args.EF, args.batch_size)
    return exp_name

def main(args):

    total_step = 100//args.EF

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # prepare checkpoints and log folders
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # initialize dataset
    if args.dataset == 'visda':
        args.data_dir = os.path.join(args.data_dir, 'visda')
        src_data = Visda_Dataset(root=args.data_dir, partition='train', label_flag=None)

    elif args.dataset == 'office':
        args.data_dir = os.path.join(args.data_dir, 'Office')
        src_data = Office_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_name,
                              target=args.target_name)

    elif args.dataset == 'home':
        args.data_dir = os.path.join(args.data_dir, 'OfficeHome')
        src_data = Home_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_name,
                              target=args.target_name)
    elif args.dataset == 'visda18':
        args.data_dir = os.path.join(args.data_dir, 'visda18')
        src_data = Visda18_Dataset(root=args.data_dir, partition='train', label_flag=None)
    else:
        print('Unknown dataset!')

    args.class_name = src_data.class_name
    args.num_class = src_data.num_class

    # number of each class
    args.alpha = src_data.alpha
    # setting experiment name

    args.experiment = set_exp_name(args)
    logger = Logger(args)

    # Phase 1
    src_trainer = SRCModelTrainer(args=args, data=src_data, logger=logger)
    model = src_trainer.train(epochs=args.pretrain_epoch)


    if args.eval_only:
        # initialize GNN
        del src_trainer
        # step to specify
        step_to_eval = 10
        trainer = ModelTrainer(args=args, data=src_data, model=model, gnn_model=None, step=step_to_eval, label_flag=None, v=None,
                                logger=logger)
        preds, labels = trainer.evaluate()
        
        draw_reliability_graph(preds, labels, args.experiment)

    else:
        # Phase 2
        pred_y, pred_score, pred_acc, pred_ent, pred_std = src_trainer.estimate_label()
        selected_idx, new_pred_y, new_pred_acc = src_trainer.select_top_data(pred_y, pred_score, pred_acc, pred_ent, pred_std)
        

        label_flag, data = src_trainer.generate_new_train_data(selected_idx, new_pred_y, new_pred_acc)
        # initialize GNN
        gnn_model = None
        del src_trainer


        # step = 1
        for step in range(1, total_step):

            print("This is {}-th step".format(step))

            trainer = ModelTrainer(args=args, data=data, model=model, gnn_model=gnn_model, step=step, label_flag=label_flag, v=selected_idx,
                                    logger=logger)

            # train the model
            # step_size = 15 + step//2
            if step == 1:
                num_epoch = 15
            else:
                num_epoch = 10 + step * 3
            model, gnn_model = trainer.train(step, epochs=num_epoch)

            # pseudo_label
            pred_y, pred_score, pred_acc, pred_ent, pred_std = trainer.estimate_label()

            # select data from target to source
            selected_idx = trainer.select_top_data(pred_y, pred_score, pred_ent, pred_std)

            # add new data
            label_flag, data = trainer.generate_new_train_data(selected_idx, pred_y, pred_acc)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source-free Progressive Graph Learning for Open-set Domain Adaptation')
    # set up dataset & backbone embedding
    dataset = 'visda18'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--graph_off', type=bool, default=True)
    parser.add_argument('--center_loss', type=bool, default=False)


    parser.add_argument('-a', '--arch', type=str, default='res')
    parser.add_argument('--root_path', type=str, default='./utils/', metavar='B',
                        help='root dir')
    parser.add_argument('--pretrain_resume', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--eval_only', type=bool, default=False)

    # set up path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data/'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'new_logs'))
    parser.add_argument('--checkpoints_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'checkpoints'))


    parser.add_argument('--pretrain_epoch', type=int, default=12)
    parser.add_argument('--tune_epoch', type=int, default=4)
    # verbose setting
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--log_epoch', type=int, default=4)

    if dataset == 'office':
        parser.add_argument('--source_name', type=str, default='D')
        parser.add_argument('--target_name', type=str, default='W')

    elif dataset == 'home':
        parser.add_argument('--source_name', type=str, default='R')
        parser.add_argument('--target_name', type=str, default='A')

    parser.add_argument('--eval_log_step', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=1500)

    # hyper-parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.1)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--EF', type=int, default=20)
    parser.add_argument('--loss', type=str, default='nll', choices=['nll', 'focal', 'smooth'])
    parser.add_argument('--ranking', type=str, default='logits', choices=['entropy', 'logits', 'uncertainty'])


    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # GNN parameters
    parser.add_argument('--in_features', type=int, default=2048)
    if dataset == 'home':
        parser.add_argument('--node_features', type=int, default=512)
        parser.add_argument('--edge_features', type=int, default=512)
    else:
        parser.add_argument('--node_features', type=int, default=1024)
        parser.add_argument('--edge_features', type=int, default=1024)

    parser.add_argument('--num_layers', type=int, default=1)

    #tsne
    parser.add_argument('--visualization', type=bool, default=False)
    #Discrminator
    parser.add_argument('--discriminator', type=bool, default=False)
    parser.add_argument('--adv_coeff', type=float, default=0.4)

    #GNN hyper-parameters
    parser.add_argument('--node_loss', type=float, default=0.3)

    parser.add_argument('--diverse_loss', type=float, default=3.0)
    main(parser.parse_args())

