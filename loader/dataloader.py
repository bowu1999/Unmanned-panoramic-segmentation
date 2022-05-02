import os
import numpy as np
import pandas as pd
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

class Getdata(torch.utils.data.Dataset):
    def __init__(self, data_root, crop_size, mode = "train"):
        '''
        crop_size: (h, w)
        is_train: True或False
        data_root: 数据集主目录名
        '''
        # 数据类型
        dtypes = ["train","test","val"]
        assert mode in dtypes, "dtype must in [train,test,val]"
        if mode == "train":
            self.root = os.path.join(data_root,"train")
            self.label_root = os.path.join(data_root,"train_labels")
        elif mode == "test":
            self.root = os.path.join(data_root,"test")
            self.label_root = os.path.join(data_root,"test_labels")
        elif mode == "val":
            self.root = os.path.join(data_root,"val")
            self.label_root = os.path.join(data_root,"val_labels")
        # 类别信息文件
        self.classes_root = os.path.join(data_root, "class_dict.csv")
        # 读取类别信息
        data = pd.read_csv(path, sep=',',header='infer')
        self.colormap = np.array(data.loc[:, ['r', 'b', 'g']]).tolist()
        self.classes = list(data["name"])
        self.colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
        for i, colormap in enumerate(self.colormap):
            self.colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        # 数据列表
        images = os.listdir(self.root)
        labels = os.listdir(self.label_root)
        # 将数据转换为tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # 过滤掉小于模型输入大小的图像
        self.images, self.labels = self.filter(images, labels, crop_size)
        # 图片大小
        self.crop_size = crop_size
        print('Read ' + str(len(self.images)) + ' valid examples')

    def filter(self, images,labels,crop_size):
        '''
        过滤掉尺寸小于crop_size的图片
        '''
        image_set = []
        label_set = []
        for i in range(len(images)):
            if (Image.open(os.path.join(self.root,images[i])).size[1] >= crop_size[0] and
                Image.open(os.path.join(self.root,images[i])).size[0] >= crop_size[1]):
                image_set.append(images[i])
                label_set.append(labels[i])
        return image_set,label_set
    
    def rand_crop(self, image, label, height, width):
        """
        随机切割为规定大小
        """
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(height, width))

        image = transforms.functional.crop(image, i, j, h, w)
        label = transforms.functional.crop(label, i, j, h, w)

        return image, label
    
    def __getitem__(self, idx):
        '''
        获取数据
        '''
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.open(os.path.join(self.root,image)).convert('RGB')
        label = Image.open(os.path.join(self.label_root,label)).convert('RGB')
        image, label = self.rand_crop(image, label, *self.crop_size)
        image = self.to_tensor(image)
        label = self.label_indices(label)

        return image, label  # float32 tensor, uint8 tensor

    def __len__(self):
        '''
        数据总量
        '''
        return len(self.images)
    
    def label_indices(self,colormap):
        """
        标签转换
        """
        colormap = np.array(colormap).astype('int32')
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return self.colormap2label[idx]
    
    def label2image(self,pred):
        colormap = torch.tensor(self.colormap, device="cpu")
        X = pred.long()
        return colormap[X, :]
    
    def show(self,key):
        """
        图片展示
        """
        image = self.images[key]
        label = self.labels[key]
        image = Image.open(os.path.join(self.root,image)).convert('RGB')
        label = Image.open(os.path.join(self.label_root,label)).convert('RGB')
        plt.subplot(121)
        plt.imshow(image), plt.axis('off')
        plt.subplot(122)
        plt.imshow(label), plt.axis('off')
        plt.show()