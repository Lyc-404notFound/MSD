import torch
from torch import nn
from torch import functional as F


class MSDLoss(nn.Module):
    def __init__(self,temp):
        super().__init__()
        self.loss1 = CELoss()
        self.loss2 = KDLoss(temp=temp)
        self.loss3 = FeatureLoss()

    def forward(self,features,outputs,labels,alpha,beta):
        loss1 = self.loss1(outputs,labels)
        loss2 = self.loss2(outputs)
        loss3 = self.loss3(features)
        loss = loss1+alpha*loss2+beta*loss3
        return loss


class KDLoss(nn.Module):
    def __init__(self,temp):
        super().__init__()
        self.temp = temp
        self.criterion = nn.KLDivLoss()
    def forward(self,outputs):
        distilled_outputs = []
        for output in outputs:
            output = torch.exp(output/self.temp)
            output_sum = torch.sum(output,dim=1,keepdim=True)
            output = output/output_sum
            distilled_outputs.append(output)
        loss2 = 0
        for i in range(len(outputs)-1):
            for j in range(len(outputs)-i-1):
                loss2 += self.criterion(distilled_outputs[i].log(),distilled_outputs[j+i+1])
        return loss2

class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(size_average=True,reduce=True)
    def forward(self,features):
        loss3 = 0
        for i in range(len(features)-1):
            loss3 += self.criterion(features[i],features[-1])
        return loss3

class CELoss(nn.Module):
    """
    outputs:共有5个outputs
    labels：只有1个labels
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=True,reduce=True)
    def forward(self,outputs,labels):
        loss1 = 0
        for i in range(len(outputs)):
            loss1 += self.criterion(outputs[i],labels)
        return loss1


