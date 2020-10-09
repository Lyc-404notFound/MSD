from EfficientNet.utils import get_model_params
from EfficientNet.EfficientNet import EfficientNet
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn


def train():
    batch_size = 32
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_dataset = datasets.CIFAR100(root='./dataset', train=True, transform=transform, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    model = EfficientNet.from_pretrained(model_name='efficientnet-b0',weights_path='./pretrainedModel/efficientnet-b0-355c32eb.pth',num_classes=100)
    loss_func = nn.CrossEntropyLoss(size_average=True,reduce=True)
    model = model.cuda()
    loss_func = loss_func.cuda()

    learning_rate = 0.01
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,verbose=True)

    model.train()
    for i,data in train_loader:
        imgs = data[0].cuda()
        labels = data[1].cuda()
        logits = model(imgs)
        softmax = torch.nn.functional.softmax(logits,dim=1)
        loss = loss_func(softmax,labels)

