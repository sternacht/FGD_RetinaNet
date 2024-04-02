import argparse
import collections
import os
import shutil
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from retinanet import coco_eval
from retinanet import csv_eval

from models import FGD_resnet50, FGD_resnet101
# assert torch.__version__.split('.')[0] == '1'

# print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('-d', '--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('-p', '--coco_path', help='Path to COCO directory', default=r'D:\master\code\FGD_implement\coco2017')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('-s', '--save_dir', help='Path to model saving dir, root dir is ./result', default='tmp')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('-m', '--train_mode', help='Training model with FGD, default training with only retinanet', type=str, default='default')
    parser.add_argument('-pt', '--pretrain_model', help='Pretrained model weight path', default=None)
    parser.add_argument('-bs', '--batch_size', help='Training batch size',type=int, default=4)
    parser.add_argument('-f', '--force_overlap', help='Overlap old dir with same name without confirm', action="store_true")
    parser.add_argument('-t', '--teacher', help='teacher model (Resnet101) path for FGD method. Only neccessary when train mode is FGD')
    parser = parser.parse_args(args)

    torch.backends.cudnn.benchmark=True
    
    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
    

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.batch_size, collate_fn=collater, batch_sampler=sampler, pin_memory=True)

    # if dataset_val is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #     dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
        print('creating model retinanet w/ resnet18')
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
        print('creating model retinanet w/ resnet34')
    elif parser.depth == 50:
        if parser.train_mode == 'FGD':
            retinanet = FGD_resnet50(num_classes=dataset_train.num_classes(), teacher_path=parser.teacher, pretrained=True)
            print('creating model retinanet w/ resnet50 and FGD')
        else:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
            print('creating model retinanet w/ resnet50')
    elif parser.depth == 101:
        if parser.train_mode == 'FGD':
            retinanet = FGD_resnet101(num_classes=dataset_train.num_classes(), teacher_path=parser.teacher, pretrained=True)
            print('creating model retinanet w/ resnet101 and FGD')
        else:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
            print('creating model retinanet w/ resnet101')
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
        print('creating model retinanet w/ resnet152')
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    if parser.pretrain_model is not None:
        try:
            retinanet.load_state_dict(torch.load(parser.pretrain_model).state_dict())
        except FileNotFoundError:
            raise FileNotFoundError(f' Pretrain model {parser.pretrain_model} is not found.')
        except TypeError:
            raise TypeError('Pretrained model weight is not fit')
        except Exception as e:
            print(e)
            exit()

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True
    init_lr = 1e-5
    optimizer = optim.Adam(retinanet.parameters(), lr=init_lr, weight_decay=0.000005)
    # optimizer = optim.SGD(retinanet.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[240000,320000], gamma=0.1)

    loss_hist = collections.deque(maxlen=100)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    save_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_path = os.path.join(save_root, parser.save_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        if not parser.force_overlap:
            print(f"Dir {parser.save_dir} is exist already. The old data will be removed.")
            is_continue = input("Enter y(Y) to continue or else to leave: ")
            if not (is_continue == 'y' or is_continue == 'Y'):
                exit()
        shutil.rmtree(save_path)
        os.mkdir(save_path)

    # cmd record
    cmd = f'python train.py -d {parser.dataset} --depth {parser.depth} -s {parser.save_dir} -e {parser.epochs} -bs {parser.batch_size} -m {parser.train_mode} -f '
    if parser.train_mode == 'FGD':
        cmd += f'-t {parser.teacher}'
    elif parser.pretrain_model is not None:
        cmd += f'-pt {parser.pretrain_model}'
    with open(os.path.join(save_path, 'cmd.txt'), 'a') as f:
        f.write(cmd)

    writer = SummaryWriter(os.path.join(save_path, 'history'))
    device = torch.device('cuda')

    # FGD hyper parameters
    alpha_fgd=1e-3
    beta_fgd=5e-4
    gamma_fgd=1e-3
    lambda_fgd=5e-6

    iter_counting = 14*len(dataloader_train)
    for iter in range(iter_counting):
        scheduler.step()

    for epoch_num in range(parser.epochs):
        optimizer.zero_grad()
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        with tqdm(total=len(dataloader_train)) as pbar:

            for iter_num, data in enumerate(dataloader_train):
                
                if torch.cuda.is_available():
                    losses = retinanet([data['img'].to(device, non_blocking=True), data['annot'].to(device, non_blocking=True)])
                else:
                    losses = retinanet([data['img'], data['annot']])

                if parser.train_mode == 'FGD':
                    classification_loss, regression_loss, global_loss, fg_feature_loss, bg_feature_loss, attention_loss = losses
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    original_loss = classification_loss + regression_loss

                    fg_feature_loss = alpha_fgd * fg_feature_loss
                    bg_feature_loss = beta_fgd * bg_feature_loss
                    attention_loss = gamma_fgd * attention_loss
                    global_loss = lambda_fgd * global_loss

                    loss =  original_loss +\
                            fg_feature_loss +\
                            bg_feature_loss +\
                            attention_loss +\
                            global_loss
                    loss_hist.append([float(classification_loss), float(regression_loss), float(fg_feature_loss), 
                                    float(bg_feature_loss), float(attention_loss), float(global_loss)])
                else:
                    classification_loss, regression_loss = losses
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    original_loss = classification_loss + regression_loss

                    loss = original_loss
                    loss_hist.append([float(classification_loss), float(regression_loss)])

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()
                optimizer.zero_grad()

                # if (iter_num+1)%(10000) == 0:
                    # scheduler.step(np.mean(epoch_loss))
                    # writer.add_scalar(f'loss/Cls', np.mean(epoch_loss),(iter_counting%100))
                    # epoch_loss = []

                epoch_loss.append(float(loss))
                mean_loss = np.mean(loss_hist,axis=0)

                if parser.train_mode == 'FGD':
                    if iter_num%100 == 0:
                        # write Class, Regression and Tol loss
                        writer.add_scalar(f'loss/Cls', mean_loss[0], iter_counting)
                        writer.add_scalar(f'loss/Reg', mean_loss[1], iter_counting)
                        writer.add_scalar(f'loss/Tol', loss, iter_counting)
                        #write FG, BG, Attention and Global loss
                        writer.add_scalar(f'loss/FG',  mean_loss[2], iter_counting)
                        writer.add_scalar(f'loss/BG',  mean_loss[3], iter_counting)
                        writer.add_scalar(f'loss/Att', mean_loss[4], iter_counting)
                        writer.add_scalar(f'loss/Glo', mean_loss[5], iter_counting)
                        iter_counting += 1
                
                    pbar.set_description(f'E:{epoch_num} | '
                                         f'Iter:{iter_num+1}/{len(dataloader_train)} | '
                                         f'Cls:{float(classification_loss):0>.4f} | '
                                         f'Reg:{float(regression_loss):0>.4f} | '
                                         f'Fg:{float(fg_feature_loss):0>.4f} | '
                                         f'Bg:{float(bg_feature_loss):0>.4f} | '
                                         f'Att:{float(attention_loss):0>.4f} | '
                                         f'Glo:{float(global_loss):0>.4f} | '
                                         f'Tol:{float(loss):0>.4f} | '
                                         f'lr:{scheduler.get_last_lr()[0] if epoch_num>0 else init_lr}')

                else:
                    if iter_num%100 == 0:
                        # write Class, Regression and Tol loss
                        writer.add_scalar(f'loss/Cls', mean_loss[0], iter_counting)
                        writer.add_scalar(f'loss/Reg', mean_loss[1], iter_counting)
                        writer.add_scalar(f'loss/Tol', loss, iter_counting)
                        iter_counting += 1
                    pbar.set_description(f'E:{epoch_num} | '
                                         f'Iter:{iter_num+1}/{len(dataloader_train)} | '
                                         f'Cls:{float(classification_loss):0>.4f} | '
                                         f'Reg:{float(regression_loss):0>.4f} | '
                                         f'Tol:{float(loss):0>.4f} | '
                                         f'lr:{scheduler.get_last_lr()[0] if epoch_num>0 else init_lr}')
                pbar.update(1)
                del classification_loss
                del regression_loss
                
        torch.save(retinanet.module, os.path.join('./results', parser.save_dir, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)))
        
        print('Evaluating dataset')

        if parser.dataset == 'coco':
            coco_eval.evaluate_coco(dataset_val, retinanet, epoch_num, writer=writer)
            
        elif parser.dataset == 'csv' and parser.csv_val is not None:
            mAP = csv_eval.evaluate(dataset_val, retinanet)
        # update scheduler if loss not decrease, only work for ReduceLROnPlateau scheduler
        scheduler.step(np.mean(epoch_loss))
        # optimizer.step()
        # scheduler.step()
    retinanet.eval()

    torch.save(retinanet, os.path.join('./results', parser.save_dir, 'model_final.pt'))


if __name__ == '__main__':
    main()
# python train.py --dataset coco --save_dir teacher_model --depth 101 --epochs 20 --pretrain_model .\results\teacher_model\coco_retinanet_11.pt