import torch.nn.functional as F
import torch.optim as optim
from models import resnet,senet,densenet
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
def main(batch_size,lr,multi_gpus,weight):
    train_loader, val_loader, test_loader = get_data(batch_size)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print('模型开始导入')

    # if multi_gpus:
    device_ids = [0, 1]
    # 创建模型
    model =senet.se_resnet50()
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

    # 加载模型

    pretained_model = torch.load(weight)
    model.load_state_dict(pretained_model)


    # model =senet.se_resnet50()
    # model = model.cuda()
    # if multi_gpus:
    #     model=torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(weight))



    print('模型导入成功')

    #     device = torch.device(0,1)
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model,device_ids = [0, 1])
    #         # ,
    # else:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.apply(inplace_relu)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)       #step_size：多少个周期后学习率发生改变  gamma:学习率如何改变

    print('模型开始训练')
    trainer = Trainer(model, optimizer, F.cross_entropy)

    mode, loss, correct, precision,recall,auc_value,test_loss_list, test_acc_list, test_data_len = trainer.test(test_loader)
#     # trainer.loop(2, train_loader, test_loader, scheduler)
    print(mode, loss, correct, test_loss_list, test_acc_list, test_data_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--multi-gpus", type=bool, default=True)
    parser.add_argument("--weight", type=str, default='runs/exp15/weights/model_epoch_10.pkl')
    args = parser.parse_args()
    main(args.batchsize, args.lr,args.multi_gpus,args.weight)
