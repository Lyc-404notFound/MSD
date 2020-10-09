import torch
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from torch.utils import data
import os
from PIL import Image
import numpy as np
from torchvision import transforms



classes = {}
index = 1

transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=tuple(np.array([125.3, 123.0, 113.9])/ 255.0),std = tuple(np.array([63.0, 62.1, 66.7]) / 255.0))
    ]
)

class VOC2007(data.Dataset):
    def __init__(self,imgsRoot,labelsRoot,transforms=None):
        imgs = os.listdir(imgsRoot)
        self.imgs = [os.path.join(imgsRoot,img) for img in  imgs]
        labels = os.listdir(labelsRoot)
        self.labels = [os.path.join(labelsRoot, label) for label in labels]
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        annotations = parseXML(self.labels[index])
        annotationsArray = convert2Array(annotations)
        sample = {'img':img,'annot':annotationsArray}
        return sample
    def __len__(self):
        return len(self.imgs)

def parseXML(path):
    xml_desc = ET.ElementTree(file=path)
    root = xml_desc.getroot()
    ObjectSet = root.findall('object')
    annotations = []
    global index,classes
    for Object in ObjectSet:
        annotation = {}
        annotation['class'] = Object.find('name').text
        if annotation['class'] not in classes.keys():
            classes[annotation['class']] = index
            index += 1;
        BndBox = Object.find('bndbox')
        annotation['xmin'] = int(float(BndBox.find('xmin').text))
        annotation['ymin'] = int(float(BndBox.find('ymin').text))
        annotation['xmax'] = int(float(BndBox.find('xmax').text))
        annotation['ymax'] = int(float(BndBox.find('ymax').text))
        annotations.append(annotation)

    return annotations

def convert2Array(annotations):
    annos = np.zeros((0,5))
    for obj in annotations:
        anno = np.zeros((1,5))
        #print(anno)
        # print(obj['xmin'])
        anno[0,0] = int(obj['xmin'])
        anno[0,1] = int(obj['ymin'])
        anno[0,2] = int(obj['xmax'])
        anno[0,3] = int(obj['ymax'])
        anno[0,4] = classes[obj['class']]
        annos = np.append(annos,anno.copy(),axis=0)
    return annos
def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:

        imgs.append(sample['img'])
        targets.append(torch.FloatTensor(sample['annot']))
    return torch.stack(imgs, 0), targets
# dataset = VOC2007(imgsRoot='../dataset/VOC2007/JPEGImages',labelsRoot='../dataset/VOC2007/Annotations',transforms=transforms)
# # print(dataset[0])
# data_iter = data.DataLoader(dataset=dataset,batch_size=1,shuffle=False,drop_last=True,collate_fn=detection_collate,pin_memory=True)
# for i,data in enumerate(data_iter):
#     if i == 1:
#         break
#     print(data[0].size())
#     label = data[1][0]
#     print(label.size())

