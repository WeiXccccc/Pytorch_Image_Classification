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
def main(batch_size,lr):
    train_loader, val_loader, test_loader = get_data(batch_size)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print('模型开始导入')

    model =senet.se_resnet50()
    model=torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('runs/exp11/weights/model_epoch_2.pkl'))
    model = model.cuda()
    print(model)

    print('模型导入成功')

    # model = model.cuda()
#这段代码一般写在读取数据之前，torch.device代表将torch.Tensor分配到的设备的对象。
# torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号。如果设备序号不存在，则为当前设备。
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model,device_ids=[0,1])
    model.to(device)
    # models.to(device)  # 使用序号为1的GPU
#     # 这也是一个优化语句，注释之后效果可能会更好点
    model.apply(inplace_relu)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)       #step_size：多少个周期后学习率发生改变  gamma:学习率如何改变

    print('模型开始训练')
    trainer = Trainer(model, optimizer, F.cross_entropy)

    mode, loss, correct, test_loss_list, test_acc_list, test_data_len = trainer.test(test_loader)
#     # trainer.loop(2, train_loader, test_loader, scheduler)
    print(mode, loss, correct, test_loss_list, test_acc_list, test_data_len)

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("--batchsize", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()
    main(args.batchsize, args.lr)
