import torch
from torch import nn
import torch.nn.functional as F
import models
from torch.utils.data import DataLoader

import os.path as osp
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils.logger import AverageMeter as meter
from data_loader import Visda_Dataset, Office_Dataset, Home_Dataset, Visda18_Dataset
from utils.loss import FocalLoss, LabelSmoothing
from torch.distributions import Categorical
from models.component import Classifier, Discriminator
import pickle
'''
In source-free seeting:
Phase 1: Trained with Source Data Only
'''
class SRCModelTrainer():
    def __init__(self, args, data, step=0, label_flag=None, v=None, logger=None):
        self.args = args
        self.batch_size = args.batch_size
        self.data_workers = 6

        self.step = step
        self.data = data
        self.label_flag = label_flag
        self.class_name = data.class_name
        self.num_class = data.num_class
        self.num_task = args.batch_size
        self.num_task = args.batch_size
        self.num_to_select = 0

        self.model = models.create(args.arch, args)
        self.model = nn.DataParallel(self.model).cuda()

        #GNN
        if not args.graph_off:
            self.gnnModel = models.create('gnn', args)
            self.gnnModel = nn.DataParallel(self.gnnModel).cuda()
        else:
            self.classifier = Classifier(args)
            self.classifier = nn.DataParallel(self.classifier).cuda()

        self.meter = meter(args.num_class)
        self.v = v

        # CE for node classification
        if args.loss == 'focal':
            self.criterionCE = FocalLoss()
        elif args.loss == 'nll':
            self.criterionCE = nn.NLLLoss(reduction='mean')
        elif args.loss == 'smooth':
            self.criterionCE = LabelSmoothing(smoothing=0.3).cuda()

        # BCE for edge
        self.criterion = nn.BCELoss(reduction='mean')
        self.global_step = 0
        self.logger = logger
        self.val_acc = 0
        self.threshold = args.threshold

    def get_dataloader(self, dataset, training=False):

        # if self.args.visualization:
        #     data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
        #                              shuffle=training, pin_memory=True, drop_last=True)
        #     return data_loader

        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                 shuffle=training, pin_memory=True, drop_last=training)
        return data_loader

    def adjust_lr(self, epoch, step_size):
        lr = self.args.lr / (2 ** (epoch // step_size))
        for g in self.optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

        if epoch % step_size == 0:
            print("Epoch {}, current lr {}".format(epoch, lr))

    def label2edge(self, targets):

        batch_size, num_sample = targets.size()
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class - 1).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        init_edge = edge*source_edge_mask.float()

        return init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask


    def transform_shape(self, tensor):

        batch_size, num_class, other_dim = tensor.shape
        tensor = tensor.view(1, batch_size * num_class, other_dim)
        return tensor

    def train(self, step=0, epochs=1, step_size=55):
        args = self.args

        train_loader = self.get_dataloader(self.data, training=True)

        # initialize model

        # change the learning rate
        if args.arch == 'res' or 'res152':
            if args.dataset == 'visda' or args.dataset == 'office' or args.dataset == 'visda18':
                param_groups = [
                        {'params': self.model.module.CNN.parameters(), 'lr_mult': 0.01}
                    ]
                if not self.args.graph_off:
                    param_groups.append({'params': self.gnnModel.parameters(), 'lr_mult': 0.1})
                if self.args.discriminator:
                    param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.1})
            else:
                param_groups = [
                    {'params': self.model.module.CNN.parameters(), 'lr_mult': 0.05}
                ]
                if not self.args.graph_off:
                    param_groups.append({'params': self.gnnModel.parameters(), 'lr_mult': 0.8})
                if self.args.discriminator:
                    param_groups.append({'params': self.discriminator.parameters(), 'lr_mult': 0.8})

            args.in_features = 2048

        elif args.arch == 'vgg':
            param_groups = [
                {'params': self.model.module.extractor.parameters(), 'lr_mult': 1}
            ]
            if not self.args.graph_off:
                param_groups.append({'params': self.gnnModel.parameters(), 'lr_mult': 1})
            args.in_features = 4096

        self.optimizer = torch.optim.Adam(params=param_groups,
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.model.train()
        if not args.graph_off:
            self.gnnModel.train()
        self.meter.reset()

        for epoch in range(epochs):
            self.adjust_lr(epoch, step_size)

            with tqdm(total=len(train_loader)) as pbar:
                for i, inputs in enumerate(train_loader):

                    images = Variable(inputs[0], requires_grad=False).cuda()
                    targets = Variable(inputs[1]).cuda()

                    # random source part
                    targets_DT = targets[:, args.num_class - 1:].reshape(-1)
                    targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)


                    init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)

                    # extract backbone features
                    features = self.model(images)
                    features = self.transform_shape(features)

                    # feed into graph networks
                    if self.args.graph_off:
                        node_logits = self.classifier(features)
                    else:
                        edge_logits, node_logits = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                                                             target_mask=target_edge_mask)

                    # compute edge loss
                    norm_node_logits = F.softmax(node_logits[-1], dim=-1).unsqueeze(0)

                    if args.loss == 'nll':
                        source_node_loss = self.criterionCE(torch.log(norm_node_logits[source_node_mask, :] + 1e-5),
                                                            targets.masked_select(source_node_mask))


                    elif args.loss == 'focal':
                        source_node_loss = self.criterionCE(norm_node_logits[source_node_mask, :],
                                                            targets.masked_select(source_node_mask))
                    elif args.loss == 'smooth':
                        source_node_loss = self.criterionCE(norm_node_logits[source_node_mask, :],
                                                            targets.masked_select(source_node_mask))
                    loss = args.node_loss * source_node_loss

                    if not args.graph_off:
                        full_edge_loss = [self.criterion(edge_logit.masked_select(source_edge_mask),
                                                         init_edge.masked_select(source_edge_mask)) for edge_logit in
                                          edge_logits]

                        edge_loss = 0
                        for l in range(args.num_layers - 1):
                            edge_loss += full_edge_loss[l] * 0.5
                        edge_loss += full_edge_loss[-1] * 1
                        loss += edge_loss


                    node_pred = norm_node_logits[source_node_mask, :].detach().cpu().max(1)[1]
                    node_prec = node_pred.eq(targets.masked_select(source_node_mask).detach().cpu()).double().mean()

                    # # Only for debugging
                    # target_labels = Variable(inputs[2]).cuda()
                    # target_labels = self.transform_shape(target_labels.unsqueeze(-1)).view(-1)
                    # if target_node_mask.any():
                    #
                    #     target_pred = norm_node_logits[target_node_mask, :].max(1)[1]
                    #
                    #     # <unlabeled> data mask
                    #     pseudo_label_mask = ~torch.eq(targets_DT, args.num_class).detach().cpu()
                    #
                    #     # remove unk for calculation
                    #     unk_label_mask = torch.eq(target_labels[~pseudo_label_mask], args.num_class - 1).detach().cpu()
                    #
                    #     # only predict on <unlabeled> data
                    #     target_prec = target_pred.eq(target_labels[~pseudo_label_mask]).double().data.cpu()
                    #
                    #     # update prec calculation on <unlabeled> data
                    #     self.meter.update(target_labels[~pseudo_label_mask].detach().cpu().view(-1).numpy(),
                    #                       target_prec.numpy())
                    #
                    #     # For pseudo_labeled data, remove unk data for prec calculation
                    #     pseudo_unk_mask = torch.eq(target_labels[pseudo_label_mask], args.num_class - 1).detach().cpu()
                    #     pseudo_prec = torch.eq(target_labels[pseudo_label_mask], targets_DT[pseudo_label_mask]).double()
                    #     if True in pseudo_unk_mask:
                    #         self.meter.update(target_labels[pseudo_label_mask].detach().cpu().masked_select(~pseudo_unk_mask).view(-1).numpy(),
                    #                           pseudo_prec.detach().cpu().masked_select(~pseudo_unk_mask).view(-1).numpy())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.logger.global_step += 1

                    # if self.args.discriminator:
                    #     self.logger.log_scalar('train/domain_loss', domain_loss, self.logger.global_step)
                    self.logger.log_scalar('train/node_prec', node_prec, self.logger.global_step)
                    self.logger.log_scalar('train/edge_loss', 0, self.logger.global_step)
                    # self.logger.log_scalar('train/OS_star', self.meter.avg[:-1].mean(), self.logger.global_step)
                    # self.logger.log_scalar('train/OS', self.meter.avg.mean(), self.logger.global_step)
                    pbar.update()

            if (epoch + 1) % args.log_epoch == 0:
                print('---- Start Epoch {} Training --------'.format(epoch))
                # for k in range(args.num_class - 1):
                #     print('Target {} Precision: {:.3f}'.format(args.class_name[k], self.meter.avg[k]))

                print('Step: {} | {}; Epoch: {}\t'
                      'Training Loss {:.3f}\t'
                      'Training Prec {:.3%}\t'
                      # 'Target Prec {:.3%}\t'
                      .format(self.logger.global_step, len(train_loader), epoch, loss.data.cpu().numpy(),
                              node_prec.data.cpu().numpy()))
                self.meter.reset()

        # save model
        states = {'model': self.model.state_dict(),
                  # 'graph': self.gnnModel.state_dict(),
                  'iteration': self.logger.global_step,
                  'val_acc': node_prec,
                  'optimizer': self.optimizer.state_dict()}
        torch.save(states, osp.join(args.checkpoints_dir, 'SRC_{}_step_{}.pth.tar'.format(args.experiment, step)))
        self.meter.reset()
        return self.model

    def estimate_label(self):
        args = self.args
        print('label estimation...')
        if args.dataset == 'visda':
            test_data = Visda_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag, target_ratio=self.step * args.EF / 100)
        elif args.dataset == 'office':
            test_data = Office_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag,
                                       source=args.source_name, target=args.target_name, target_ratio=self.step * args.EF / 100)
        elif args.dataset == 'home':
            test_data = Home_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag, source=args.source_name,
                              target=args.target_name, target_ratio=self.step * args.EF / 100)
        elif args.dataset == 'visda18':
            test_data = Visda18_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag, target_ratio=self.step * args.EF / 100)

        self.meter.reset()
        # append labels and scores for target samples
        pred_labels = []
        pred_scores = []
        real_labels = []
        pred_entropy = []
        pred_std = []
        target_loader = self.get_dataloader(test_data, training=False)
        self.model.eval()

        # features_to_save = []
        # self.gnnModel.eval()
        with tqdm(total=len(target_loader)) as pbar:
            for i, (images, targets, target_labels, split) in enumerate(target_loader):

                images = Variable(images, requires_grad=False).cuda()
                targets = Variable(targets).cuda()
                targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)

                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)

                # extract backbone features
                features = self.model(images)
                features = self.transform_shape(features)
                torch.cuda.empty_cache()
                # feed into graph networks
                # edge_logits, node_logits = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                #                                          target_mask=target_edge_mask)
                node_logits = self.classifier(features)
                # features_to_save.append(features.detach().cpu().numpy())
                del features
                norm_node_logits = F.softmax(node_logits[-1], dim=-1).unsqueeze(0)
                if args.ranking == 'entropy':
                    target_entropy = Categorical(probs = norm_node_logits[target_node_mask, :]).entropy()
                    pred_entropy.append(target_entropy.cpu().detach())
                elif args.ranking == 'uncertainty':
                    target_std = torch.std(norm_node_logits[target_node_mask, :], dim=0)
                    pred_std.append(target_std)
                target_score, target_pred = norm_node_logits[target_node_mask, :].max(1)

                # only for debugging
                target_labels = Variable(target_labels).cuda()
                target_labels = self.transform_shape(target_labels.unsqueeze(-1)).view(-1)
                pred = target_pred.detach().cpu()
                target_prec = pred.eq(target_labels.detach().cpu()).double()

                self.meter.update(
                    target_labels.detach().cpu().view(-1).data.cpu().numpy(),
                    target_prec.numpy())

                pred_labels.append(target_pred.cpu().detach())
                pred_scores.append(target_score.cpu().detach())
                real_labels.append(target_labels.cpu().detach())
                
                if i % self.args.log_step == 0:
                    print('Step: {} | {}; \t'
                          'OS Prec {:.3%}\t'
                          .format(i, len(target_loader),
                                  self.meter.avg.mean()))

                pbar.update()

        # with open("15_target_features.pkl", "wb") as f:
        #     pickle.dump(features_to_save, f)
        
        # print("target features are saved.")
        
        # # with open("target_labels.pkl", "wb") as f:
        # #     pickle.dump(real_labels, f)
        
        # print("target labels are saved.")

        # weights = self.classifier.module.classifier.weight.detach().cpu().numpy()
        # with open("15_classifier_weights.pkl", "wb") as f:
        #     pickle.dump(weights, f)

        # print("class weights are saved.")

        
        pred_labels = torch.cat(pred_labels)
        pred_scores = torch.cat(pred_scores)
        
        real_labels = torch.cat(real_labels)

        self.model.train()
        # self.gnnModel.train()
        self.num_to_select = int(len(target_loader) * self.args.batch_size * (self.args.num_class - 1) * self.args.EF / 100)

        if args.ranking == 'entropy':
            pred_entropy = torch.cat(pred_entropy)
            return pred_labels.data.cpu().numpy(), pred_scores.data.cpu().numpy(), real_labels.data.cpu().numpy(), pred_entropy.data.cpu().numpy(), None
        elif args.ranking == 'uncertainty':
            pred_std = torch.cat(pred_std)
            return pred_labels.data.cpu().numpy(), pred_score.data.cpu().numpy(), real_label.data.cpu().numpy(), None, pred_std.data.cpu().numpy()
        else: 
            return pred_labels.data.cpu().numpy(), pred_scores.data.cpu().numpy(), real_labels.data.cpu().numpy(), None, None

    def select_top_data(self, pred_label, pred_score, pred_entropy=None, pred_std=None):
        # remark samples if needs pseudo labels based on classification confidence
        highest_conf_recorder = np.zeros((self.num_class, ))
        if self.v is None:
            self.v = np.zeros(len(pred_score))
        unselected_idx = np.where(self.v == 0)[0]

        if len(unselected_idx) < self.num_to_select:
            self.num_to_select = len(unselected_idx)

        if pred_entropy is not None:
            index = np.argsort(-pred_score[unselected_idx] + 0.4*pred_entropy[unselected_idx])
        else:
            index = np.argsort(-pred_score[unselected_idx])
        index_orig = unselected_idx[index]
        num_pos = int(self.num_to_select * self.threshold / (self.num_class - 1))
        class_recorder = np.ones((self.num_class - 1, )) * num_pos
        num_neg = self.num_to_select - int(num_pos * (self.num_class - 1))
        i = 0
        while class_recorder.any():
            if class_recorder[pred_label[index_orig[i]]] > 0:
                self.v[index_orig[i]] = 1
                class_recorder[pred_label[index_orig[i]]] -= 1
                if class_recorder[pred_label[index_orig[i]]] == 0:
                    highest_conf_recorder[pred_label[index_orig[i]]] = pred_score[index_orig[i]]
            i += 1
            if i >= len(index_orig):
                break
        for i in range(1, num_neg + 1):
            self.v[index_orig[-i]] = -1
        # record the threshhold for the unk class
        highest_conf_recorder[-1] = pred_score[index_orig[-i]]
        for i in range(self.num_class):
            print("Pseudo Label for class {} is threshholded by {}".format(self.class_name[i], highest_conf_recorder[i]))
        return self.v


    def generate_new_train_data(self, sel_idx, pred_y, real_label):
        # create the new dataset merged with pseudo labels
        assert len(sel_idx) == len(pred_y)
        new_label_flag = []
        pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
        for i, flag in enumerate(sel_idx):
            if i >= len(real_label):
                break
            if flag > 0:
                new_label_flag.append(pred_y[i])
                pos_total += 1
                if real_label[i] == pred_y[i]:
                    pos_correct += 1
            elif flag < 0:
                # assign the <unk> pseudo label
                new_label_flag.append(self.args.num_class - 1)
                pred_y[i] = self.args.num_class - 1
                neg_total += 1
                if real_label[i] == self.args.num_class - 1:
                    neg_correct += 1
            else:
                new_label_flag.append(self.args.num_class)


        self.meter.reset()
        self.meter.update(real_label, (pred_y == real_label).astype(int))

        for k in range(self.args.num_class):
            print('Target {} Precision: {:.3f}'.format(self.args.class_name[k], self.meter.avg[k]))

        for k in range(self.num_class):
            self.logger.log_scalar('test/' + self.args.class_name[k], self.meter.avg[k], self.step)
        self.logger.log_scalar('test/ALL', self.meter.sum.sum() / self.meter.count.sum(), self.step)
        self.logger.log_scalar('test/OS_star', self.meter.avg[:-1].mean(), self.step)
        self.logger.log_scalar('test/OS', self.meter.avg.mean(), self.step)
        self.logger.log_scalar('test/H-score', (2 * self.meter.avg[-1] * self.meter.avg[:-1].mean()) /
                               (self.meter.avg[-1] + self.meter.avg[:-1].mean()), self.step)

        print('Node predictions: OS accuracy = {:0.4f}, OS* accuracy = {:0.4f}'.format(self.meter.avg.mean(), self.meter.avg[:-1].mean()))

        correct = pos_correct + neg_correct
        total = pos_total + neg_total
        acc = correct / total
        pos_acc = pos_correct / pos_total
        neg_acc = neg_correct / neg_total
        new_label_flag = torch.tensor(new_label_flag)

        # update source data
        if self.args.dataset == 'visda':
            new_data = Visda_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                     target_ratio=(self.step + 1) * self.args.EF / 100)

        elif self.args.dataset == 'office':
            new_data = Office_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                       source=self.args.source_name, target=self.args.target_name,
                                      target_ratio=(self.step + 1) * self.args.EF / 100)

        elif self.args.dataset == 'home':
            new_data = Home_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                    source=self.args.source_name, target=self.args.target_name,
                                    target_ratio=(self.step + 1) * self.args.EF / 100)
        elif self.args.dataset == 'visda18':
            new_data = Visda18_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                     target_ratio=(self.step + 1) * self.args.EF / 100)

        print('selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}'.format(correct, total, acc))
        print('positive data: {} of {} is correct, accuracy: {:0.4f}'.format(pos_correct, pos_total, pos_acc))
        print('negative data: {} of {} is correct, accuracy: {:0.4f}'.format(neg_correct, neg_total, neg_acc))
        return new_label_flag, new_data

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes, dtype=torch.long)[class_idx]

    def load_model_weight(self, path):
        print('loading weight')
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.gnnModel.load_state_dict(state['graph'])

    def label2edge_gt(self, targets):
        '''
        creat initial edge map and edge mask for unlabeled targets
        '''
        batch_size, num_sample = targets.size()
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class - 1).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(
            torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        # unlabeled flag

        return (edge*source_edge_mask.float())

    def extract_feature(self):
        print('Feature extracting...')
        self.meter.reset()
        # append labels and scores for target samples
        vgg_features_target = []
        node_features_target = []
        labels = []
        overall_split = []
        target_loader = self.get_dataloader(self.data, training=False)
        self.model.eval()
        self.gnnModel.eval()
        num_correct = 0
        skip_flag = self.args.visualization
        with tqdm(total=len(target_loader)) as pbar:
            for i, (images, targets, target_labels, _, split) in enumerate(target_loader):

                # for debugging
                # if i > 100:
                #     break
                images = Variable(images, requires_grad=False).cuda()
                targets = Variable(targets).cuda()

                # only for debugging
                # target_labels = Variable(target_labels).cuda()

                targets = self.transform_shape(targets.unsqueeze(-1)).squeeze(-1)
                target_labels = self.transform_shape(target_labels.unsqueeze(-1)).squeeze(-1).cuda()
                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)
                # gt_edge = self.label2edge_gt(target_labels)
                # extract backbone features
                features = self.model(images)
                features = self.transform_shape(features)

                # feed into graph networks
                edge_logits, node_feat = self.gnnModel(init_node_feat=features, init_edge_feat=init_edge,
                                                         target_mask=target_edge_mask)
                vgg_features_target.append(features.data.cpu())
                #####heat map only
                # temp = np.array(edge_logits[0].data.cpu()) * 4
                # ax = sns.heatmap(temp.squeeze(), vmax=1)#
                # cbar = ax.collections[0].colorbar
                # # here set the labelsize by 20
                # cbar.ax.tick_params(labelsize=17)
                # plt.savefig('heat/' + str(i) + '.png')
                # plt.close()
                ###########
                node_features_target.append(node_feat[-1].data.cpu())
                labels.append(target_labels.data.cpu())
                overall_split.append(split)
                if skip_flag and i > 50:
                    break

                pbar.update(n=self.num_class*2)

        return vgg_features_target, node_features_target, labels, overall_split









