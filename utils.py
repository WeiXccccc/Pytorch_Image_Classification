from pathlib import Path
import torch
import torch.backends.cudnn
from tqdm import tqdm
import gc
import numpy as np
from matplotlib import pyplot as plt
import time
import glob
import re
plt.switch_backend('agg')

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
time_list = []
timestart = time.perf_counter()


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        # self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_reslut = str(Save_Dir())

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        for data, target in tqdm(data_loader, ncols=80):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)

            # 在InceptionV3中需要修改为outputs,aux2,aux1 = net(inputs)
            # outputs,aux2,aux1 = self.models(data)      不会修改
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            # scheduler.step()     PyTorch 1.1.0 之前， scheduler.step() 应该在 optimizer.step() 之前调用。现在这么做则会跳过学习率更新的第一个值。
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        mode = "train" if is_train else "test"
        print(">>>[{}] loss: {:.4f}/accuracy: {:.4f} ".format(mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset) ))
        return mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset)

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            mode, loss, correct = self._iteration(data_loader)
            # return mode, loss, correct
            train_loss_list.append(loss)
            train_acc_list.append(correct)
            train_data_len = len(data_loader.dataset)
            return mode, loss, correct, train_loss_list, train_acc_list, train_data_len


    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            mode, loss, correct = self._iteration(data_loader, is_train=False)
            #  by YXY
            test_loss_list.append(loss)
            test_acc_list.append(correct)
            test_data_len = len(data_loader.dataset)
            return mode, loss, correct, test_loss_list, test_acc_list, test_data_len
            # return mode, loss, correct


    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            # 添加一个根据epoch改变学习率的代码，好像很难实现
            epochstart = time.perf_counter()

            print("epochs: {}".format(ep))
            # save statistics into txt file
            self.save_statistic_train(*((ep,) + self.train(train_data)))

            if scheduler is not None:
                scheduler.step()
                # scheduler.step(ep)
            self.save_statistic_test(*((ep,) + self.test(test_data)))
            if scheduler is not None:
                scheduler.step()
                # scheduler.step(ep)
            if ep % self.save_freq == 0:
                self.save_model(ep)
            gc.collect()
            elapsed = (time.perf_counter() - epochstart)
            time_list.append(elapsed)
            print('第 %d 周期训练和测试用的Time used %.6f s \n\n' %(ep, elapsed))
            save_time(ep, elapsed,self.save_reslut)
            if ep % 2 == 0:     #画出训练集和测试集的图形       根据之前训练集和测试集的训练loss和准确度进行绘图，在这里通过plt.savefig将其保存到了本地
                x = np.linspace(0, len(train_loss_list), len(test_loss_list))
                plt.plot(x, train_loss_list, label="train_loss", linewidth=1.5)
                plt.plot(x, test_loss_list, label="test_loss", linewidth=1.5)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                # plt.show()
                plt.savefig(self.save_reslut+'/2loss.jpg')
                # print('Picture has Finish,but you can alse use visual/viz.py to read state.txt and write it ,Which is word by YXY')
                plt.clf()

                x = np.linspace(0, len(train_loss_list), len(test_loss_list))
                plt.plot(x, train_acc_list, label="train_acc", linewidth=1.5)
                plt.plot(x, test_acc_list, label="test_acc", linewidth=1.5)
                plt.xlabel("epoch")
                plt.ylabel("acc")
                plt.legend()
                # plt.show()
                plt.savefig(self.save_reslut+'/2acc.jpg')

        timesum = (time.perf_counter() - timestart)
        print('\nThe total time is %.6f  s' %(timesum))
        save_time_end(111, timesum,self.save_reslut)



    def save_model(self, epoch, **kwargs):
        if self.save_reslut is not None:
            model_out_path = Path(self.save_reslut+'/weights')
            # state = {"epoch": epoch, "net_state_dict": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            # torch.save(self.model.state_dict(), model_out_path / "model_epoch_{}.ckpt".format(epoch))
            torch.save(self.model.state_dict(), model_out_path/"model_epoch_{}.pkl".format(epoch))

            # torch.save(models.state_dict(), './checkpoint/VGG19_Cats_Dogs_hc.pth')



    def save_statistic(self, epoch, mode, loss, accuracy):
        with open(self.save_reslut+"/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")
    def save_statistic_train(self, epoch, mode, loss, accuracy, train_loss_list, train_acc_list, train_data_len):
        with open(self.save_reslut+"/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")
        # torch.save(mode.state_dict(), "my_model.pth")  # 只保存模型的参数
        # torch.save(models, "my_model.pth")  # 保存整个模型
    def save_statistic_test(self, epoch, mode, loss, accuracy, test_loss_list, test_acc_list, test_data_len):
        with open(self.save_reslut+"/state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")




def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def save_time(epoch, elapsed,save_time_path):
    with open(save_time_path+"/state_time.txt", "a", encoding="utf-8") as f:
        f.write(str({"每个epoch的": epoch, "elapsed": elapsed}))
        f.write("\n")
def save_time_end(epoch, elapsed,save_time_end_path):
    with open(save_time_end_path+"/state_time.txt", "a", encoding="utf-8") as f:
        f.write(str({"最后epoch的": epoch, "elapsed": elapsed}))
        f.write("\n")

def Save_Dir(save_dir='runs/exp'):
    save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/exp', mkdir=True)  # increment save_dir
    print("本次训练结果保存在"+str(save_dir))
    return save_dir



