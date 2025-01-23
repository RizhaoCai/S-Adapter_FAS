import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.utils import AverageMeter
from .network import build_net

pd.set_option('display.max_columns', None)

import logging

from data.transforms import VisualTransform, get_augmentation_transforms

from test import metric_report_from_dict

# try:
# from torch.utils.tensorboard import SummaryWriter
# except:
from tensorboardX import SummaryWriter

from .get_dataset import get_image_dataset_from_list


class Trainer():
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """

        self.config = config
        self.global_step = 1
        self.start_epoch = 1

        # Training control config
        self.epochs = self.config.TRAIN.EPOCHS
        self.batch_size = self.config.DATA.BATCH_SIZE

        self.counter = 0

        #  # Meanless at this version
        self.epochs = self.config.TRAIN.EPOCHS
        self.val_freq = config.TRAIN.VAL_FREQ
        #
        # # Network config

        self.val_metrcis = {
            'HTER@0.5': 1.0,
            'EER': 1.0,
            'MIN_HTER': 1.0,
            'AUC': 0
        }

        # Optimizer config
        self.momentum = self.config.TRAIN.MOMENTUM
        self.init_lr = self.config.TRAIN.INIT_LR
        self.lr_patience = self.config.TRAIN.LR_PATIENCE
        self.train_patience = self.config.TRAIN.PATIENCE


        self.train_mode = True

        kwargs = {
            'conv_type': self.config.MODEL.CONV,
            'num_classes': self.config.MODEL.NUM_CLASSES,
            'cdc_theta': self.config.MODEL.CDC_THETA
        }
        self.network = build_net(arch_name=config.MODEL.ARCH, pretrained=config.MODEL.IMAGENET_PRETRAIN, **kwargs)


        self.test_dataset = None

        self.loss = torch.nn.CrossEntropyLoss()

    def init_weight(self, ckpt_path):
        logging.info("[*] Initialize model weight from {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.network.load_state_dict(ckpt['model_state'])


    def get_dataset(self, data_file_list, transform):
        datasets = []
        if self.config.MODEL.NUM_CLASSES>2:
            if_binary = False
        else:
            if_binary = True
        for data_list_path in data_file_list:
            real_data_set, fake_dataset = get_image_dataset_from_list(data_list_path, transform, if_binary_label=if_binary)
            datasets.append(real_data_set)
            datasets.append(fake_dataset)

        return torch.utils.data.ConcatDataset(datasets), datasets

    def get_train_dataset(self, data_file_list, transform):
        datasets = []
        if self.config.MODEL.NUM_CLASSES > 2:
            if_binary = False
        else:
            if_binary = True

        real_dataset_list, fake_dataset_list = [], []
        for data_list_path in data_file_list:
            real_dataset, fake_dataset = get_image_dataset_from_list(data_list_path, transform,
                                                                      if_binary_label=if_binary)
            real_dataset_list.append(real_dataset)
            fake_dataset_list.append(fake_dataset)

        return  real_dataset_list, fake_dataset_list

    def extract_data_from_loader(self, data_loader):
        try:
            out = data_loader['iterator'].next()
            img_tensor, labels = out[1], out[2]
        except:
            data_loader['iterator'] = iter(data_loader['loader'])
            out = data_loader['iterator'].next()
            img_tensor, labels = out[1], out[2]

        spoofing_labels = labels['spoofing_label']
        return img_tensor, spoofing_labels

    def get_dataloader(self):
        config = self.config

        train_batch_size = config.DATA.BATCH_SIZE
        val_batch_size = config.TEST.BATCH_SIZE
        test_batch_size = config.TEST.BATCH_SIZE
        num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS

        dataset_root_dir = config.DATA.ROOT_DIR
        dataset_subdir = config.DATA.SUB_DIR  # 'EXT0.2'
        dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)

        test_data_transform = VisualTransform(config)

        if not self.train_mode:
            assert config.DATA.TEST, "Please provide at least a data_list"
            test_dataset, _ = self.get_dataset(config.DATA.TEST, test_data_transform)
            self.test_data_loader = torch.utils.data.DataLoader(test_dataset, test_batch_size, num_workers=num_workers,
                                                                shuffle=True, drop_last=True)
            self.test_dataset = test_dataset
            return self.test_data_loader

        else:
            assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"
            aug_transform = get_augmentation_transforms(config)
            train_data_transform = VisualTransform(config, aug_transform)
            train_real_dataset_list, train_fake_dataset_list = self.get_train_dataset(config.DATA.TRAIN, train_data_transform)




            self.train_real_dataloader_list = [
                torch.utils.data.DataLoader(train_dataset, train_batch_size //len(train_real_dataset_list)//2,
                                            num_workers=num_workers //len(train_dataset)//2,
                                            shuffle=True, pin_memory=True, drop_last=True) for train_dataset in
                train_real_dataset_list
            ]

            self.train_fake_dataloader_list = [
                torch.utils.data.DataLoader(train_dataset, train_batch_size // len(train_real_dataset_list) // 2,
                                            num_workers=num_workers // len(train_dataset) // 2,
                                            shuffle=True, pin_memory=True, drop_last=True) for train_dataset in
                train_fake_dataset_list
            ]

            self.train_real_dataloader_list = [ {'loader':x, 'iterator': iter(x)} for x in
                                                self.train_real_dataloader_list
                                                ]
            self.train_fake_dataloader_list = [{'loader': x, 'iterator': iter(x)} for x in
                                               self.train_fake_dataloader_list
                                               ]

            assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
            val_dataset, _ = self.get_dataset(config.DATA.VAL, test_data_transform)
            self.val_data_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size, num_workers=num_workers,
                                                               shuffle=False, pin_memory=True, drop_last=True)

            return self.train_real_dataloader_list, self.val_data_loader

    def train(self, ):


        if self.config.TRAIN.INIT and os.path.exists(self.config.TRAIN.INIT):
            self.init_weight(self.config.TRAIN.INIT)

        if self.config.CUDA:
            self.network.cuda()

        if self.config.MODEL.FIX_BACKBONE:
            for name, p in self.network.named_parameters():
                if 'adapter' in name or 'head' in name:
                    p.requires_grad = True
                    # import pdb; pdb.set_trace()
                else:
                    p.requires_grad = False

        if not self.config.MODEL.FIX_NORM:
            for name, p in self.network.named_parameters():
                if 'norm' in name:
                    p.requires_grad = True
                    # import pdb; pdb.set_trace()


        # Set up optimizer
        if self.config.TRAIN.OPTIM.TYPE == 'SGD':
            logging.info('Setting: Using SGD Optimizer')
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.init_lr,
            )

        elif self.config.TRAIN.OPTIM.TYPE == 'Adam':
            logging.info('Setting: Using Adam Optimizer')
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.init_lr,
            )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.TRAIN.EPOCHS)

        # Set up datasets and data loaders
        self.get_dataloader()
        # train_data_loader_list = self.train_data_loader_list
        val_data_loader = self.val_data_loader
        tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        self.num_train = 100000
        self.num_valid = len(val_data_loader) * self.config.DATA.BATCH_SIZE
        logging.info("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid))

        if self.config.TRAIN.RESUME and os.path.exists(self.config.TRAIN.RESUME):
            logging.info("Resume=True.")
            self.load_checkpoint(self.config.TRAIN.RESUME)

        if self.config.CUDA:
            logging.info("Number of GPUs: {}".format(torch.cuda.device_count()))
            self.network = torch.nn.DataParallel(self.network)

        pbar = tqdm(range(self.config.TRAIN.MAX_ITER + 1), ncols=120)
        train_loss = AverageMeter()
        self.network.train()


        self.mlp_token_out = []
        def hook_feature(module, input, output):
            self.mlp_token_out.append(output)

        hook_handle = self.network.module.blocks[11].mlp.register_forward_hook(hook_feature)


        for global_step in pbar:
            if self.tensorboard:
                self.tensorboard.add_scalar('lr', self.init_lr, self.global_step)
            #logging.info('\nEpoch: {}/{} - LR: {:.6f}'.format(
            #    epoch, self.epochs, self.init_lr))



            # ------------------------------------ Validation ----------------------------------------------------------
            if (global_step+1) % self.config.TRAIN.VAL_FREQ == 0:
                train_loss_avg = train_loss.avg
                self.global_step = global_step
                logging.info("Global step at {} | Avg BC loss = {} ".format(global_step, str(train_loss_avg)))
                # evaluate on validation set'
                if train_loss.avg < 0.4:
                    with torch.no_grad():
                        hook_handle.remove()
                        val_output = self.validate(global_step, val_data_loader)
                        self.network.train()
                        hook_handle = self.network.module.blocks[11].mlp.register_forward_hook(hook_feature)

                    if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                        logging.info("Save models")
                        self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                        self.val_metrcis['AUC'] = val_output['AUC']
                        self.save_checkpoint(
                            {'epoch': global_step,
                             'val_metrics': self.val_metrcis,
                             'global_step': self.global_step,
                             'model_state': self.network.module.state_dict(),
                             'optim_state': self.optimizer.state_dict(),
                             }
                        )

                    logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                             100 * self.val_metrcis['AUC']))
                    train_loss = AverageMeter()
                    logging.info("Reset Training Loss AverageMeter")
                else:
                    logging.info("Avg training loss larger than threshold, skip validation")



            # -----------------------------------Data Loading ----------------------------------------------------------
            # Set to train model
            # Load data
            real_input_list = []
            real_label_list = []
            fake_input_list = []
            fake_label_list = []
            num_domains = len(self.train_real_dataloader_list)
            for i in range(num_domains):
                real_dataloader, fake_dataloader = self.train_real_dataloader_list[i], self.train_fake_dataloader_list[i]

                real_input_tensor, real_spoofing_label = self.extract_data_from_loader(real_dataloader)
                fake_input_tensor, fake_spoofing_label = self.extract_data_from_loader(fake_dataloader)

                real_input_list.append(real_input_tensor)
                real_label_list.append(real_spoofing_label)

                fake_input_list.append(fake_input_tensor)
                fake_label_list.append(fake_spoofing_label)

            all_real_input_tensor = torch.cat(real_input_list, 0)
            all_real_labels = torch.cat(real_label_list, 0)

            all_fake_input_tensor = torch.cat(fake_input_list, 0)
            all_fake_labels = torch.cat(fake_label_list, 0)


            # -----------------------------------------------------------------------------------------------------------
            def get_gram_matrix(feat_map):
                B, C, H, W = feat_map.size()
                feat_map = feat_map.view(B, C, -1)
                gram_matrix = torch.bmm(feat_map, feat_map.permute(0,2,1)) / C / H / W
                return gram_matrix
            real_cls_out = self.inference(all_real_input_tensor.cuda())

            # real
            if 'base' in self.config.MODEL.ARCH:
                reduction_dim = 768
                # model = vit_base_patch16_224(pretrained, num_classes=num_classes)  # todo

            if 'large' in self.config.MODEL.ARCH:
                reduction_dim = 1024
                # model = vit_large_patch16_224(pretrained, num_classes=num_classes)  # todo

            if 'small' in self.config.MODEL.ARCH:
                reduction_dim = 384
                # model = vit_small_patch16_224(pretrained, num_classes=num_classes)  # todo

            if 'tiny' in self.config.MODEL.ARCH:
                reduction_dim = 192
                # model = vit_tiny_patch16_224(pretrained, num_classes=num_classes)  # todo
            #
            if num_domains == 3:
                B = real_cls_out.shape[0]
                tokens_out = self.mlp_token_out.pop()
                tokens_out = tokens_out[:, 1:, :].reshape(-1, 14, 14, reduction_dim).permute(0, 3, 1, 2)
                domain_seg = B//num_domains
                # Domain 1
                tokens_out_1 = tokens_out[:domain_seg,:,:,:] # (B//3, reduction_dim, 14, 14)
                tokens_out_2 = tokens_out[domain_seg:2*domain_seg, :, :, :] # (B//3, reduction_dim, 14, 14)
                tokens_out_3 = tokens_out[domain_seg*2:, :, :, :] # (B//3, reduction_dim, 14, 14)


                gram_matrix_1 = get_gram_matrix(tokens_out_1)
                gram_matrix_2 = get_gram_matrix(tokens_out_2)
                gram_matrix_3 = get_gram_matrix(tokens_out_3)

                gram_loss = (torch.norm(gram_matrix_1-gram_matrix_2, p='fro')+torch.norm(gram_matrix_1-gram_matrix_3, p='fro') + torch.norm(gram_matrix_3-gram_matrix_2, p='fro'))/3
            elif num_domains == 2:
                B = real_cls_out.shape[0]
                tokens_out = self.mlp_token_out.pop()
                tokens_out = tokens_out[:, 1:, :].reshape(-1, 14, 14, reduction_dim).permute(0, 3, 1, 2)
                domain_seg = B // num_domains
                # Domain 1
                tokens_out_1 = tokens_out[:domain_seg, :, :, :]  # (B//3, reduction_dim, 14, 14)
                tokens_out_2 = tokens_out[domain_seg:2 * domain_seg, :, :, :]  # (B//3, reduction_dim, 14, 14)
                #tokens_out_3 = tokens_out[domain_seg * 2:, :, :, :]  # (B//3, reduction_dim, 14, 14)

                gram_matrix_1 = get_gram_matrix(tokens_out_1)
                gram_matrix_2 = get_gram_matrix(tokens_out_2)
                #gram_matrix_3 = get_gram_matrix(tokens_out_3)

                gram_loss = torch.norm(gram_matrix_1 - gram_matrix_2, p='fro')





            fake_cls_out = self.inference(all_fake_input_tensor.cuda())
            # compute losses for differentiable modules
            _ = self.mlp_token_out.pop()
            del _

            all_cls_out = torch.cat([real_cls_out,fake_cls_out], 0)


            all_labels = torch.cat([all_real_labels,all_fake_labels], dim=0).cuda()


            bc_loss = self.loss(all_cls_out, all_labels)

            # import pdb; pdb.set_trace()
            # compute gradients and update SGD
            gram_loss_weight = self.config.TRAIN.W_GRAM
            loss = bc_loss + gram_loss_weight*gram_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            pbar.set_description(
                (
                    " total loss={:.4f}|bc_loss=={:.4f}|gram_loss={:.4f}".format(loss.item(),bc_loss.item(), gram_loss.item()
                                                 )
                )

            )
            pbar.update(1)
            train_loss.update(bc_loss.item(), 1)
            # log to tensorboard
            if self.tensorboard:
                self.tensorboard.add_scalar('loss/train_total', loss.item(), self.global_step)
                self.tensorboard.add_scalar('loss/bc_total', bc_loss.item(), self.global_step)

            self.global_step += 1


            #self.lr_scheduler.step()




    def validate(self, epoch, val_data_loader):
        val_results = self.test(val_data_loader)
        val_loss = val_results['avg_loss']
        scores_gt_dict = val_results['scores_gt']
        scores_pred_dict = val_results['scores_pred']
        # log to tensorboard
        if self.tensorboard:
            self.tensorboard.add_scalar('loss/val_total', val_loss, self.global_step)

        frame_metric_dict, video_metric_dict = metric_report_from_dict(scores_pred_dict, scores_gt_dict, 0.5)
        df_frame = pd.DataFrame(frame_metric_dict, index=[0])
        df_video = pd.DataFrame(video_metric_dict, index=[0])

        logging.info("Frame level metrics: \n" + str(df_frame))
        logging.info("Video level metrics: \n" + str(df_video))

        return frame_metric_dict

    def test(self, test_data_loader):
        if test_data_loader is None:
            batch_size = self.config.TEST.BATCH_SIZE
            num_workers = self.config.DATA.NUM_WORKERS

            test_data_transform = VisualTransform(self.config)
            dataset = self.get_dataset(self.config.DATA.TEST, test_data_transform)
            test_data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers,
                                                                shuffle=False, drop_last=False)
        avg_test_loss = AverageMeter()
        scores_pred_dict = {}
        spoofing_label_gt_dict = {}
        self.network.eval()

        with torch.no_grad():
            for data in tqdm(test_data_loader, ncols=80):
                network_input, target, video_ids = data[1], data[2], data[3]
                x = network_input.cuda()
                # import pdb; pdb.set_trace()
                output_prob = self.inference(x)
                test_loss = self.loss(output_prob, target['spoofing_label'].cuda())
                pred_score = self._get_score_from_prob(output_prob)
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                gt_dict, pred_dict = self._collect_scores_from_loader(spoofing_label_gt_dict, scores_pred_dict,
                                                                      target['spoofing_label'].numpy(), pred_score,
                                                                      video_ids
                                                                      )

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
        }
        return test_results

    def _collect_scores_from_loader(self, gt_dict, pred_dict, ground_truths, pred_scores, video_ids):
        batch_size = ground_truths.shape[0]

        for i in range(batch_size):
            video_name = video_ids[i]
            if video_name not in pred_dict.keys():
                pred_dict[video_name] = list()
            if video_name not in gt_dict.keys():
                gt_dict[video_name] = list()

            pred_dict[video_name] = np.append(pred_dict[video_name], pred_scores[i])
            gt_dict[video_name] = np.append(gt_dict[video_name], ground_truths[i])

        return gt_dict, pred_dict

    def save_checkpoint(self, state):

        ckpt_dir = os.path.join(self.config.OUTPUT_DIR, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        epoch = state['epoch']
        if self.config.TRAIN.SAVE_BEST:
            filename = 'best.ckpt'.format(epoch)
        else:
            filename = 'epoch_{}.ckpt'.format(epoch)
        ckpt_path = os.path.join(ckpt_dir, filename)
        logging.info("[*] Saving model to {}".format(ckpt_path))
        torch.save(state, ckpt_path)

    def load_checkpoint(self, ckpt_path):

        logging.info("[*] Loading model from {}".format(ckpt_path))

        ckpt = torch.load(ckpt_path)
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.valid_metric = ckpt['val_metrics']
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.network.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )

    def inference(self, *args, **kargs):
        """
            Input images
            Output prob and scores
        """
        output_prob = self.network(*args, **kargs)  # By default: a binary classifier network
        return output_prob


    def _get_score_from_prob(self, output_prob):
        output_scores = torch.softmax(output_prob, 1)
        output_scores = 1-output_scores.cpu().numpy()[:, 0] # Probability to be spoofing
        return output_scores

    def load_batch_data(self):
        pass

