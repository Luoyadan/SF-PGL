from __future__ import print_function, absolute_import
import os
import argparse
import random
import numpy as np
# torch-related packages
import torch
import matplotlib.pyplot as plt
from utils.visualization import visualize_TSNE
torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

from datetime import datetime
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

# data
from data_loader import Visda_Dataset, Office_Dataset, Home_Dataset, Visda18_Dataset
from model_trainer_new import ModelTrainer
from src_model_trainer_new import SRCModelTrainer
from utils.logger import Logger

def main(args):
    start_time = datetime.now()
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


    if not args.visualization:
            # Phase 1
            src_trainer = SRCModelTrainer(args=args, data=src_data, logger=logger)
            model = src_trainer.train(epochs=args.pretrain_epoch, step_size=args.pretrain_epoch)
            # Phase 2
            pred_y, pred_score, pred_acc, pred_ent, pred_std = src_trainer.estimate_label()
            selected_idx, new_pred_y, new_pred_acc = src_trainer.select_top_data(pred_y, pred_score, pred_acc, pred_ent, pred_std)
            

            label_flag, data = src_trainer.generate_new_train_data(selected_idx, new_pred_y, new_pred_acc)
            # initialize GNN
            gnn_model = None
            del src_trainer

            end_time = datetime.now()
            print('Finished source pre-training: {}'.format(end_time - start_time))
            # step = 1
            for step in range(1, total_step):

                print("This is {}-th step".format(step))

                trainer = ModelTrainer(args=args, data=data, model=model, gnn_model=gnn_model, step=step, label_flag=label_flag, v=selected_idx,
                                        logger=logger)

                # train the model
                # step_size = 15 + step//2

                args.log_epoch = 1 + step//2
                if step == 1 or step == 2:
                    num_epoch = 1
                else:
                    num_epoch = 2 + step // 2
                model, gnn_model = trainer.train(step, epochs=num_epoch, step_size=args.log_epoch)

                # pseudo_label
                pred_y, pred_score, pred_acc, pred_ent, pred_std = trainer.estimate_label()

                # select data from target to source
                selected_idx = trainer.select_top_data(pred_y, pred_score, pred_ent, pred_std)

                # add new data
                label_flag, data = trainer.generate_new_train_data(selected_idx, pred_y, pred_acc)

    else:
        # evaluation only
        if os.path.exists('./vis.pickle'):
            with open('./vis.pickle', 'rb') as f:
                data = pickle.load(f)
            node_feat = data['node_feat']
            target_labels = data['target_labels']
            split = data['split']
            visualize_TSNE(node_feat, target_labels, args.num_class, args, split)

            plt.savefig('./node_tsne.pdf', dpi=500)
            print('successfully drawed and saved.')
        else:
            step_to_eval = args.step_to_eval
            # Load model from Phase 1
            src_model = SRCModelTrainer(args=args, data=src_data, step=step_to_eval, logger=logger)
            
            # initialize GNN
        
            trainer = ModelTrainer(args=args, data=src_data, model=src_model, gnn_model=None, step=step_to_eval, label_flag=None, v=None,
                                    logger=logger)
            _, node_feat, target_labels, split = trainer.extract_feature()
            vis_data = {'node_feat':node_feat, 'target_labels':target_labels, 'split':split}
            with open('./vis.pickle', 'wb') as f:
                pickle.dump(vis_data, f)
            print('successfully saved vis data.')
    end_time = datetime.now()
    print('Finished all training: {}'.format(end_time - start_time))
            
    



def set_exp_name(args):
    exp_name = 'D-{}'.format(args.dataset)
    if args.dataset == 'office' or args.dataset == 'home':
        exp_name += '_src-{}_tar-{}'.format(args.source_name, args.target_name)
    exp_name += '_A-{}'.format(args.arch)
    exp_name += '_L-{}'.format(args.num_layers)
    exp_name += '_E-{}_B-{}'.format(args.EF, args.batch_size)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source-free Progressive Graph Learning for Open-set Domain Adaptation on Visda-17')
    # set up dataset & backbone embedding
    dataset = 'visda'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--graph_off', type=bool, default=True)
    parser.add_argument('-a', '--arch', type=str, default='res', choices=['res', 'res152', 'vgg'])
    parser.add_argument('--root_path', type=str, default='./utils/', metavar='B',
                        help='root dir')
    parser.add_argument('--pretrain_resume', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--step_to_eval', type=int, default=19)

    # set up path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data/'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'new_logs'))
    parser.add_argument('--checkpoints_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'checkpoints'))


    parser.add_argument('--pretrain_epoch', type=int, default=2)
    parser.add_argument('--tune_epoch', type=int, default=0)
    # verbose setting
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--log_epoch', type=int, default=2)

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
    parser.add_argument('--threshold', type=float, default=0.7)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--EF', type=int, default=5)
    parser.add_argument('--loss', type=str, default='nll', choices=['nll', 'focal'])
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
    parser.add_argument('--adv_coeff', type=float, default=0)

    parser.add_argument('--center_loss', type=bool, default=False)
    #GNN hyper-parameters
    parser.add_argument('--node_loss', type=float, default=0.3)
    parser.add_argument('--diverse_loss', type=float, default=0)
    parser.add_argument('--entropy_loss', type=float, default=0.5)
    main(parser.parse_args())

