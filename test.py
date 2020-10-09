from EfficientNet.utils import get_model_params
import torch,os,time
from LYCNet import LYCNet
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
from tqdm.autonotebook import tqdm
from loss.MSDLoss import MSDLoss
from math import cos,pi
import numpy as np
from utils.logging import get_logger
import datetime


def train():
    #获取模型的参数，args是efficientNet网络的结构参数，params模型的超参数
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger = get_logger(f'logging/{now}.txt')
    args,params = get_model_params('efficientnet-b0',None)
    # print(args,params)
    model = LYCNet(args,params)
    #加载预训练的模型
    model_dict = model.state_dict()
    pretrain_dict = torch.load('./pretrainedModel/efficientnet-b0-355c32eb.pth')
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    #数据初始化
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    distill_temp = 5
    loss_func = MSDLoss(distill_temp)
    batch_size = 64
    num_epoches  = 100
    learning_rate = 0.01

    #载入数据集
    train_dataset = datasets.CIFAR100(root='./dataset',train=True,transform=transform,download=True)
    val_dataset = datasets.CIFAR100(root='./dataset',train=False,transform=transform,download=True)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True,drop_last=True)

    # progress_bar = tqdm(train_loader)
    #这里定义损失函数，temp表示蒸馏损失

    iter_num = len(train_dataset)//batch_size
    loss_func = loss_func.cuda()
    model = model.cuda()

  #定义优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,verbose=True)


    #损失函数的参数
    alpha = 0.1
    beta_begin = 0.1
    beta_end = 0.001
    logger.info('training params:sample_numbers = {},distill_temp = {},batch_size = {},num_epoches = {},learning_rate = {},alpha = {},\
    beta_begin = {},beta_end = {}'.format(len(train_dataset),distill_temp,batch_size,num_epoches,learning_rate,alpha,beta_begin,beta_end))
    # print(str(len(train_dataset))+'\n')
    model.train()
    logger.info('start training!')
    for epoch in range(num_epoches):
        time1 = time.time()
        epoch_loss = []
        beta = 0.5*(1+cos(pi*epoch/num_epoches)*(beta_begin-beta_end))+beta_end
        # print(f'alpha = {alpha}   beta = {beta}\n')
        for iter,data in enumerate(train_loader):
            imgs = data[0]
            imgs = imgs.cuda()
            # print(imgs.size())
            labels = data[1]
            labels = labels.cuda()

            optimizer.zero_grad()
            features,outputs = model(imgs)
            loss = loss_func(features,outputs,labels,alpha,beta)
            if loss ==0 or not torch.isfinite(loss):
                continue
            #计算梯度
            loss.backward()
            #更新参数
            optimizer.step()
            epoch_loss.append(float(loss))
        scheduler.step(np.mean(epoch_loss))
        # print(f'epoch:{epoch + 1}  : loss  = {np.mean(epoch_loss)}\n,cost_time = {time.time()-time1}')
        logger.info('Epoch:[{}/{}]\t time = {:.3f}\t loss = {:.5f}\t alpha = {}\t beta = {:.5f}'.format(epoch+1,num_epoches,time.time()-time1,np.mean(epoch_loss),alpha,beta))
        if epoch%10 == 0:
            torch.save(model,f'LYCNet_{epoch}epoch.path')

def test():
    # 获取模型的参数，args是efficientNet网络的结构参数，params模型的超参数
    args, params = get_model_params('efficientnet-b0', None)
    # print(args,params)
    model = LYCNet(args, params)
    # 加载预训练的模型
    # model_dict = model.state_dict()
    # pretrain_dict = torch.load('./pretrainedModel/efficientnet-b0-355c32eb.pth')
    # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(model_dict)
    # 数据初始化
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    distill_temp = 5
    loss_func = MSDLoss(distill_temp)
    batch_size = 32
    # num_epoches = 100
    # learning_rate = 0.01

    # 载入数据集
    # train_dataset = datasets.CIFAR100(root='./dataset', train=True, nsforms=transform, download=True)
    val_dataset = datasets.CIFAR100(root='./dataset', train=False, transform=transform, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    # progress_bar = tqdm(train_loader)
    # 这里定义损失函数，temp表示蒸馏损失

    iter_num = len(val_dataset) // batch_size
    loss_func = loss_func.cuda()
    model = model.cuda()

    # 定义优化器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # 损失函数的参数
    alpha = 0.1
    # beta_begin = 0.1
    # beta_end = 0.001
    beta = 0.4515244252618963
    print(str(len(val_dataset)) + '\n')
    checkpoints = [0,10,20,30,40,50,60,70,80,90]
    model.eval()
    for i,value in enumerate(checkpoints):
        model.load_state_dict(torch.load(f'LYCNet_{value}epoch.path').state_dict())
        time1 = time.time()
        epoch_loss = []
        print(f'alpha = {alpha}   beta = {beta}')
        correct_num_list = [0,0,0,0,0]
        for iter, data in enumerate(val_loader):
            imgs = data[0]
            imgs = imgs.cuda()
            # print(imgs.size())
            labels = data[1]
            labels = labels.cuda()

            # optimizer.zero_grad()
            features, outputs = model(imgs)
            correct_num_list = [x+correct_num_list[i] for i,x in enumerate(cal_correct_num(outputs,labels))]
            # print(labels)
            # print('\n--------------\n')
            # print(features[0].size())
            # print('\n--------------\n')
            # print(outputs[0].size())
            # print('\n--------------\n')
            loss = loss_func(features, outputs, labels, alpha, beta)
            # print(loss)
            if loss == 0 or not torch.isfinite(loss):
                continue
            # 计算梯度
            # loss.backward()
            # 更新参数
            # optimizer.step()

            epoch_loss.append(float(loss))
        # scheduler.step(np.mean(epoch_loss))
        print(f'checkpoint:epoch{value}  : loss = {np.mean(epoch_loss)},cost_time = {time.time() - time1},accuracy = {[x/(iter_num*batch_size) for x in correct_num_list]}')
        # if epoch % 10 == 0:
        #     torch.save(model, f'LYCNet_{epoch}epoch.path')

def save_checkpoint(model,name):
    torch.save(model.state_dict(),f'checkpoint/{name}.pth')

def cal_correct_num(outputs,labels):
    correct_num_list = []
    for output in outputs:
        logits = output.argmax(dim = 1)
        correct_num = torch.eq(logits,labels).sum()
        correct_num_list.append(correct_num)
    correct_num_list = [int(x) for x in correct_num_list]
    return correct_num_list

if __name__ == '__main__':
    train()
    # test()




