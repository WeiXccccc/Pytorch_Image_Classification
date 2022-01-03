import torch.nn.functional as F
import torch.optim as optim
from models import resnet,senet,densenet,vgg
from utils import Trainer,Save_Dir
from dataset import get_data
import argparse
import torch
import torch.backends.cudnn
import torch.nn as nn


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(batch_size,lr,epoch,multi_gpus,data_path,weight):
    train_loader, val_loader, test_loader = get_data(batch_size,root=data_path)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print('Strart Import Model...')

    model =senet.se_resnet50(num_classes=2)



    print('Import Model Success...')

    model = model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if multi_gpus:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            if weight != '':
                pretained_model = torch.load(weight)
                model.load_state_dict(pretained_model)
                print('Load weight:' + weight)
    model.to(device)

    model.apply(inplace_relu)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)       #step_size：多少个周期后学习率发生改变  gamma:学习率如何改变

    print('Start Trainning...')
    trainer = Trainer(model, optimizer, F.cross_entropy)

    trainer.loop(epoch, train_loader, test_loader, scheduler)
    print('Trainning End...')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=float, default=10)
    parser.add_argument("--multi-gpus", type=float, default=True)
    parser.add_argument("--data_path", type=str, default='D:/data_9_COVID2/')
    parser.add_argument("--weight", type=str, default='runs/exp2/weights/model_epoch_10.pkl')
    args =parser.parse_args()
    main(args.batchsize,args.lr,args.epoch,args.multi_gpus,args.data_path,args.weight)

