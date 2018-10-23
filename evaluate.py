

# import the packages
import argparse
import logging
import sys
import time
import os
import numpy as np
from torchvision.datasets import ImageFolder
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ==================================
# control input options. DO NOT CHANGE THIS PART.
from torch.optim.lr_scheduler import StepLR


def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for part 3 of project 1')
    parser.add_argument('--cuda', action='store_true', default=True,
        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')
    parser.add_argument('--load_path',default= './vgg16_best_accur.pth',type=str)
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    pargs = parser.parse_args()
    return pargs

def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

def eval_net(net, loader, logging):
    net = net.eval()
    if args.cuda:
        net = net.cuda()
    net.load_state_dict(torch.load(args.load_path,map_location='cuda'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log
    logging.info('=' * 55)
    logging.info('SUMMARY ')
    logging.info('Accuracy of the network test images: %d %%' % (
        100 * correct / total))
    logging.info('=' * 55)

# Prepare for writing logs and setting GPU. 
# DO NOT CHANGE THIS PART.
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)
logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

# DO NOT change codes above this line
# ==================================


####################################
# Transformation definition
# NOTE:
# Write the assigned transformation method here.
# Your modification of transformation should be performed on training
# set, i.e. train_transform. You can keep test_transform untouched or
# the same as the train_transform or using something else.

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


test_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                 ])

####################################

####################################
# Define training and test dataset. 
# You can make some modifications, e.g. batch_size, adding other hyperparameters
batch_size = args.batch_size
testset = ImageFolder('./release/val', transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)


# ==================================

####################################
# Define Optimizer or Scheduler
# NOTE:
# define your optimizer or scheduler below.

print('==> Building model..')
net = torchvision.models.vgg16()
####################################

# ==================================
# finish training and test the network
# and write to logs. DO NOT CHANGE THIS PART.
# train modified network
# train_net(modified, trainloader, logging, criterion, optimizer, scheduler)

# test the baseline network and modified network
# To keep the input size for training and test the same after applying Resize transform
# in modified model, here corresponding testloaders are used for baseline and modified.
# eval_net(baseline, baseline_testloader, logging, mode="baseline")
# eval_net(modified, modified_testloader, logging, mode="modified")
eval_net(net, testloader, logging)
# ==================================
