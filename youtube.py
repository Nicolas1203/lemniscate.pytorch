'''Train Youtube thumbnails with PyTorch.'''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd

import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms

import os
import argparse
import time

import models
import math

from datasets.YoutubeDataset import YoutubeDataset
from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Youtube Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=0, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('-d', '--img-dir', help="Path/to/dir/containing/images",
                    default='./data/thumbnails/')
parser.add_argument('-t', '--train', action='store_true',
                    help='Run this code in training mode on the selected data.')
parser.add_argument('--encode', action='store_true',
                    help="Run the code in encoding mode. Encode all the images in the selected folder.")


def main():
    global args, device
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.train:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2,1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = YoutubeDataset(img_dir=args.img_dir, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        ndata = trainset.__len__()

    print('==> Building model..')
    net = models.__dict__['resnet18'](low_dim=args.low_dim)
    if args.train:
        # define leminiscate
        if args.nce_k > 0:
            lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
        else:
            lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Model
    if len(args.resume)>0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.resume)
        net.load_state_dict(checkpoint['net'])
        lemniscate = checkpoint['lemniscate']
        start_epoch = checkpoint['epoch']
    
    if args.train:
        # define loss function
        if hasattr(lemniscate, 'K'):
            criterion = NCECriterion(ndata)
        else:
            criterion = nn.CrossEntropyLoss()

    net.to(device)
    if args.train:
        lemniscate.to(device)
        criterion.to(device)

    if args.train:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch, net, optimizer, trainloader, lemniscate, criterion)
            if not epoch % 10:  
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'lemniscate': lemniscate,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, f"./checkpoint/ckpt_{epoch}.t7")

    # Generate encoding for the data selected
    if args.encode:        
        encode(net, args.img_dir, args.resume)


def encode(net, img_dir, resume):
    """Encode the data from the loader and save every encoding and image path in a csv.

    Args:
        net ([type]): [description]
        img_dir: [description]
        resume: [description]
    """
    print("Loading data for encoding...")
    transform_encode = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    encode_set = YoutubeDataset(img_dir, transform=transform_encode)
    encode_loader = torch.utils.data.DataLoader(encode_set, batch_size=1, shuffle=False, num_workers=1)

    print("Encoding data...")
    if len(resume) < 2:
        raise Warning("You must specify a model to resume when running in encoding mode")
    
    all_data = []
    for inputs, _, img_path in tqdm(encode_loader):
        inputs.to(device)
        features = net(inputs)
        line = [img_path, features.tolist()[0]]
        all_data.append(line)
    df_all = pd.DataFrame(all_data, columns=['img_path', 'features'])
    df_all.to_csv(f'./data/features_{resume[:-3]}.csv')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch-80) // 40))
    print(f"Adjusted lr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(epoch, net, optimizer, trainloader, lemniscate, criterion):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, indexes, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, indexes = inputs.to(device), indexes.to(device)
        optimizer.zero_grad()

        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

if __name__ == '__main__':
    main()
