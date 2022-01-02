from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.backends.cudnn
from tqdm import tqdm
import gc
import numpy as np
from matplotlib import pyplot as plt
import time
import glob
import re
from torchmetrics import Precision,Recall,F1,AUC
from sklearn.metrics import roc_curve,auc,roc_auc_score,confusion_matrix,accuracy_score,precision_score,recall_score
import itertools

plt.switch_backend('TkAgg')

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

        self.precision_opt = Precision( average='micro',num_classes=2).cuda()
        self.recall_opt = Recall(average='micro',num_classes=2).cuda()
        self.f1_opt = F1(num_classes=2).cuda()
        # self.roc_opt = ROC(num_classes=2).cuda()


    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        precision = []
        recall = []
        F1 = []
        Test_Target = []
        Test_Pred_Conf = []
        Pred=[]
        for data, target in tqdm(data_loader, ncols=80,total=len(data_loader)):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            #### output，pred
            output = self.model(data)
            pred_= output.data.max(1)
            pred = output.data.max(1)[1]
            # 预测置信度
            pred_conf = output.data[:, 0]

            loss = self.loss_f(output,target)
            precision_value,recall_value,F1_value= self.P_R_F1(pred,target)
            precision.append(precision_value)
            recall.append(recall_value)
            F1.append(F1_value)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            mode = "train" if is_train else "test"
            if mode=='test':
                Test_Target.append(target.data.cpu().numpy())
                Test_Pred_Conf.append(pred_conf.data.cpu().numpy())
                Pred.append(pred.data.cpu().numpy())

            print(">>>[{}] loss: {:.4f}/accuracy: {:.4f} precision:{:.4f} recall: {:.4f} F1 score: {:.4f}".format(mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset),sum(precision)/len(precision),sum(recall)/len(recall),sum(F1)/len(F1)))

        return mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset),  Test_Target , Test_Pred_Conf, Pred

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            mode, loss, correct,Test_Target,Test_Pred_Conf,pred= self._iteration(data_loader)
            # return mode, loss, correct
            train_loss_list.append(loss)
            train_acc_list.append(correct)
            train_data_len = len(data_loader.dataset)

            return mode, loss, correct, train_loss_list, train_acc_list, train_data_len


    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            mode, loss, correct,Test_Target, Test_Pred_Conf,Pred= self._iteration(data_loader, is_train=False)
            Pred = [raw for raws in Pred for raw in raws]
            Test_Target = [raw for raws in Test_Target for raw in raws]
            Test_Pred_Conf = [raw for raws in Test_Pred_Conf for raw in raws]

            #Roc绘制
            auc_value = Plot_ROC(self.save_reslut,Test_Target,Test_Pred_Conf)
            # auc_ = auc(Test_Target,Test_Pred_Conf)
            #混淆矩阵计算
            cm = confusion_matrix(y_true=Test_Target, y_pred=Pred)
            # 混淆矩阵绘制
            plot_confusion_matrix(self.save_reslut,cm,['0','1'],normalize=False)

            accuracy = accuracy_score(y_true=Test_Target, y_pred=Pred)

            precision=precision_score(y_true=Test_Target, y_pred=Pred)

            recall = recall_score(y_true=Test_Target, y_pred=Pred)

            test_loss_list.append(loss)
            test_acc_list.append(correct)
            test_data_len = len(data_loader.dataset)
            return mode, loss, correct, precision,recall,auc_value,test_loss_list, test_acc_list, test_data_len


    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            # 添加一个根据epoch改变学习率的代码，好像很难实现
            epochstart = time.perf_counter()

            print("epochs: {}".format(ep))

            # save statistics into txt file
            mode, loss, correct, train_loss_list, train_acc_list, train_data_len = self.train(train_data)
            self.save_statistic_train(ep,mode, loss, correct, train_loss_list, train_acc_list, train_data_len)

            if scheduler is not None:
                scheduler.step()
                # scheduler.step(ep)
            mode, loss, correct, precision,recall,auc_value,test_loss_list, test_acc_list, test_data_len = self.test(test_data)
            self.save_statistic_test(ep,mode, loss, correct, precision,recall,auc_value,test_loss_list, test_acc_list, test_data_len)
            if scheduler is not None:
                scheduler.step()
                # scheduler.step(ep)
            if ep % self.save_freq == 0:
                self.save_model(ep)
            gc.collect()

            elapsed = (time.perf_counter() - epochstart)
            time_list.append(elapsed)

            Plot_loss(self.save_reslut,train_loss_list,test_loss_list)
            Plot_Acc(self.save_reslut,train_acc_list,test_acc_list)
        timesum = (time.perf_counter() - timestart)
        print('\nThe total time is %.6f  s' %(timesum))
        Plot(self.save_reslut)


    def P_R_F1(self,pred,target):
        precision_value = self.precision_opt(pred, target)
        recall_value = self.recall_opt(pred, target)
        F1_value = self.f1_opt(pred, target)

        return precision_value,recall_value,F1_value

    def save_model(self, epoch, **kwargs):
        if self.save_reslut is not None:
            model_out_path = Path(self.save_reslut+'/weights')
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(self.model.state_dict(), model_out_path/"model_epoch_{}.pkl".format(epoch))



    def save_statistic_train(self, epoch, mode, loss, accuracy, train_loss_list, train_acc_list, train_data_len):
        with open(self.save_reslut+"/train.log", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")

    def save_statistic_test(self, epoch,mode, loss, correct, precision,recall,auc_value,test_loss_list, test_acc_list, test_data_len):
        with open(self.save_reslut+"/test.log", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": correct,"precision": precision,"recall": recall,"auc_value": auc_value}))
            f.write("\n")

def Plot_ROC(path,Test_Target,Test_Pred_Conf):
        fpr_value, tpr_value, thresholds = roc_curve(Test_Target, Test_Pred_Conf, pos_label=0)
        auc_value = auc(fpr_value,tpr_value)
        if len(fpr_value) > 2 and len(tpr_value) > 2 and len(thresholds) > 2:
            plt.plot(fpr_value, tpr_value)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.savefig(path + '/ROC_Curve.jpg',bbox_inches='tight')
            plt.clf()
        return auc_value

def Plot_Acc(path,train_acc_list,test_acc_list):
     x = np.linspace(0, len(train_acc_list), len(test_acc_list))
     plt.plot(x, train_acc_list, label="train_acc", linewidth=1.5)
     plt.plot(x, test_acc_list, label="test_acc", linewidth=1.5)
     plt.xlabel("epoch")
     plt.ylabel("acc")
     plt.legend()
     plt.savefig(path + '/Acc.jpg',bbox_inches='tight')
     plt.clf()


def Plot_loss(path,train_loss_list,test_loss_list):
    x = np.linspace(0, len(train_loss_list), len(test_loss_list))
    plt.plot(x, train_loss_list, label="train_loss", linewidth=1.5)
    plt.plot(x, test_loss_list, label="test_loss", linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path + '/Loss.jpg',bbox_inches='tight')
    plt.clf()

# 绘制混淆矩阵
def plot_confusion_matrix(path,cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Purples):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path + '/Confusion_matrix.jpg',bbox_inches='tight')
    plt.clf()

def increment_path(path, exist_ok=False, sep='', mkdir=False):
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


class Plot:
    def __init__(self, root):
        self.path = root
        self.data = self.read_txt(self.path+'/train.log')
        self.train_data = self.get_train_data(self.data)
        print("best train accuracy:")
        print(self.get_best_val_acc(self.train_data["accuracy"]))
        self.data = self.read_txt(self.path + '/test.log')
        self.val_data = self.get_val_data(self.data)
        print("best val accuracy:")
        print(self.get_best_val_acc(self.val_data["accuracy"]))
        self.show_train(self.train_data,self.path)
        self.show_val(self.val_data,self.path)

    def read_txt(self,root):
        with open(root, "r", encoding="utf-8") as file:
            content = [eval(line.replace("\n", "")) for line in file.readlines()]
            return content

    def get_train_data(self,data):
        assert isinstance(data, list), "``data`` should be list type"
        epoch = []
        acc = []
        loss = []
        for line in data:
            if line["mode"] == "train":
                epoch.append(line["epoch"])
                acc.append(line["accuracy"])
                loss.append(line["loss"])
        return {"epoch": epoch, "accuracy": acc, "loss": loss}

    def get_val_data(self,data):
        assert isinstance(data, list), "``data`` should be ``list`` type"
        epoch = []
        acc = []
        loss = []
        for line in data:
            if line["mode"] == "test":
                epoch.append(line["epoch"])
                acc.append(line["accuracy"])
                loss.append(line["loss"])
        return {"epoch": epoch, "accuracy": acc, "loss": loss}

    def get_best_val_acc(self,data):
        assert isinstance(data, list), "``data`` must be ``list`` type"
        return max(data)

    def show_train(self,data,path):
        assert isinstance(data, dict), "``data`` should be ``dict`` type"
        _, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
        axs[0].plot(data["epoch"], data["accuracy"])
        axs[0].set_title("accuracy")
        axs[1].plot(data["epoch"], data["loss"])
        axs[1].set_title("loss")
        plt.savefig(path+'/Train_Acc_Loss_Curve.jpg',bbox_inches='tight')

    def show_val(self,data,path):
        assert isinstance(data, dict), "``data`` should be ``dict`` type"
        _, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
        axs[0].plot(data["epoch"], data["accuracy"])
        axs[0].set_title("accuracy")

        axs[1].plot(data["epoch"], data["loss"])
        axs[1].set_title("loss")
        plt.savefig(path+'/Test_Acc_Loss_Curve.jpg',bbox_inches='tight')

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
if __name__ == "__main__":
    path = r"runs/exp11"

    Plot(path)