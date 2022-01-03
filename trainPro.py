import torch.nn.functional as F
import torch.optim as optim
from models import resnet,senet,densenet,vgg
from utils import Trainer,Save_Dir
from dataset import get_data
import argparse
import torch
import torch.backends.cudnn
import torch.nn as nn
import numpy as np
import yaml
from torch.optim import SGD, Adam, lr_scheduler

def train(hyp, opt,device):
    print("2")

def main(opt):
    print('1')
    # Resume

    # DDP mode

    # Train
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=float, default=10)
    parser.add_argument("--multi-gpus", type=float, default=False)
    parser.add_argument("--data_path", type=str, default='D:/data_9_COVID2/')
    parser.add_argument("--weight", type=str, default='')
    parser.add_argument("--hyp", type=str, default='hyp.scratch.yaml')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)