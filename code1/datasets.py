import random
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

from Utils import np2tensor,preprocess_input
import os
from files import TxtFile
from copy import deepcopy
import mxnet as mx
# rgb2gray = transforms.Compose([transforms.ToPILImage(), transforms.Resize([112, 112]), transforms.Grayscale(3),
#                                transforms.ToTensor(),
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# lab_transforms = transforms.Compose([transforms.Resize([112, 112]), transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                                      ])
# hf = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
# vf = transforms.Compose([transforms.ToPILImage(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
#
# rgb2graypoints = transforms.Compose(
#     [transforms.ToTensor(), transforms.ToPILImage(), transforms.Grayscale(3), transforms.ToTensor()])
# hfpoints = transforms.Compose(
#     [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
# vfpoints = transforms.Compose(
#     [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
#
# r2g = transforms.Compose([transforms.ToPILImage(),
#                           transforms.TenCrop(39),
#                           transforms.Lambda(lambda crops: torch.stack([rgb2graypoints(crop) for crop in crops]))])
# horizontal_flip = transforms.Compose([transforms.ToPILImage(),
#                                       transforms.TenCrop(39),
#                                       transforms.Lambda(lambda crops: torch.stack([hfpoints(crop) for crop in crops]))])
# vertical_flip = transforms.Compose([transforms.ToPILImage(),
#                                     transforms.TenCrop(39),  # this is a list of PIL Images,一张图变成10张图
#                                     transforms.Lambda(lambda crops: torch.stack([vfpoints(crop) for crop in crops]))
#                                     # returns a 4D tensor
#                                     ])
# normal = transforms.Compose([transforms.ToPILImage(),
#                              transforms.TenCrop(39),
#                              transforms.Lambda(
#                                  lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
num_image = 0
class ContrastiveTrain(Dataset):
    # relationlist = ['ss', 'md', 'fs', 'ms', 'fd', 'sibs', 'bb', 'gmgd', 'gfgd']
    def __init__(self,
                 sample_path="../data/FIW/pairs/newTrain2.txt", # 训练样本路径
                 dataset_root="../data/FIW",   # 数据集根目录
                 images_size=112,        # 图片大小
                 backbone='vit',
                 relation  = None,
                 rand_mirror = True
                 ):          # 特征提取模型的骨干网络


        self.backbone=backbone
        self.dataset_root=dataset_root
        self.sample_path=sample_path
        self.images_size = images_size
        self.bias = 0                     # 样本偏移量
        self.relation = relation
        self.samples_list=self.load_samples()           # 样本列表
        self.rand_mirror=rand_mirror



    def load_samples(self):
        lines=TxtFile.read(self.sample_path)            # 读取样本路径
        t_lines = []
        p_lines = []
        for t_line in lines:
            if (t_line[3] == self.relation):
                t_lines.append(t_line)
                if (t_line[4] == '1'):
                    p_lines.append(t_line)
        lines = t_lines
        pos_lines = p_lines

        p_lines = []
        # print("train正样本对长度", len(pos_lines))

        samples=[[line[1],line[2],line[3],int(line[-1])] for line in lines]    # 样本包括两张图片路径、亲戚关系和标签

        return samples

    def __len__(self):
        # print("train长度", len(self.samples_list))
        num_image = len(self.samples_list)
        return num_image

    def set_bias(self,bias):
        self.bias=bias    # 设置样本偏移量


    # 读取图片并进行预处理，使其适合特征提取模型的输入格式
    def read_image(self, path): # 定义了一个名为 read_image 的函数，它有两个参数：self，它表示该方法所调用的类的实例；path，它是一个字符串，表示要加载的图像文件路径。
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))  # os.path.join 函数将 self.dataset_root 和 path 连接起来作为图像的完整路径，然后调用 resize 方法将图像缩放到指定的大小
        img = np.array(img, dtype=np.float)  # 将 PIL 图像对象转换为 NumPy 数组，并将数组类型转换为浮点数。
        if self.backbone == 'vit':
            img = preprocess_input(img, version=2) # 调用 preprocess_input 函数对图像进行特定的预处理，例如在使用 ResNet50 模型时，需要将图像的像素值归一化为一定范围内的值。

        if self.rand_mirror:
            # print("image randon flip")
            _rd = random.randint(0, 1)
            if _rd == 1:
                img = mx.nd.array(img)
                img = mx.ndarray.flip(data=img, axis=1)

        img = np.transpose(img, (2, 0, 1)) #将 NumPy 数组的维度顺序从 (height, width, channel) 转换为 (channel, height, width)。
        img = np2tensor(img.copy()) # 将 NumPy 数组转换为 PyTorch 张量
        return img


    def __getitem__(self, item):
        # lines = TxtFile.read(self.sample_path)  # 读取样本路径
        sample = self.samples_list[item+self.bias] # 获取指定样本
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])  # 读取两张图片
        kin_class=sample[2]   # 获取亲子关系
        label = np2tensor(np.array(int(sample[3]),dtype=np.float))  # 获取标签




        return img1,img2,kin_class,label   # 返回样本数据和标签




class ContrastiveVal(Dataset):
    def __init__(self,
                 sample_path='../data/FIW/pairs/val_choose.txt',
                 dataset_root="../data/FIW",
                 images_size=112,
                 backbone='vit',
                 relation=None
                 ):

        self.backbone=backbone
        self.dataset_root=dataset_root
        self.sample_path=sample_path
        self.images_size = images_size
        self.relation = relation
        self.samples_list=self.load_samples()


    def load_samples(self):
        lines = TxtFile.read(self.sample_path)  # 读取样本路径
        t_lines = []
        p_lines = []
        n_lines = []
        for t_line in lines:
            if (t_line[3] == self.relation):
                t_lines.append(t_line)
                if (t_line[4] == '1'):
                    p_lines.append(t_line)
                else:
                    n_lines.append(t_line)
        lines = t_lines
        pos_lines = p_lines
        neg_lines = n_lines

        # p_lines = []
        # print("val正样本对长度", len(pos_lines))
        # print("val负样本对长度", len(neg_lines))
        samples = [[line[1], line[2], line[3], int(line[-1])] for line in lines]  # 样本包括两张图片路径、亲戚关系和标签
        return samples


    def __len__(self):
        # print("val长度", len(self.samples_list))
        return len(self.samples_list)


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone=='vit':
            img = preprocess_input(img, version=2)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item):
        sample = self.samples_list[item]
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        kin_class = sample[2]
        label = np2tensor(np.array(int(sample[-1]))) # 将样本的最后一个元素（即标签）转换成一个整数，然后用 NumPy 将这个整数转换成一个 NumPy 数组，最后将这个 NumPy 数组转换成 PyTorch 张量并赋值给变量 label
        return img1,img2,kin_class,label



class ContrastiveTest(Dataset):
    def __init__(self,
                 sample_path='../data/FIW/pairs/val_choose.txt',
                 dataset_root="../data/FIW",
                 images_size=112,
                 backbone='vit',
                 relation=None
                 ):

        self.backbone = backbone
        self.dataset_root = dataset_root
        self.sample_path = sample_path
        self.images_size = images_size
        self.relation = relation
        self.samples_list = self.load_samples()

    def load_samples(self):
        lines = TxtFile.read(self.sample_path)  # 读取样本路径
        t_lines = []
        p_lines = []
        n_lines = []
        for t_line in lines:
            if (t_line[3] == self.relation):
                t_lines.append(t_line)
                if (t_line[4] == '1'):
                    p_lines.append(t_line)
                else:
                    n_lines.append(t_line)
        lines = t_lines
        pos_lines = p_lines
        neg_lines = n_lines
        # p_lines = []
        # print("test正样本对长度", len(pos_lines))
        # print("test负样本对长度", len(neg_lines))
        samples = [[line[1], line[2], line[3], int(line[-1])] for line in lines]  # 样本包括两张图片路径、亲戚关系和标签
        return samples

    def __len__(self):
        print("test长度", len(self.samples_list))
        return len(self.samples_list)

    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone == 'vit':
            img = preprocess_input(img, version=2)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item):
        sample = self.samples_list[item]
        img1, img2 = self.read_image(sample[0]), self.read_image(sample[1])
        kin_class = sample[2]
        label = np2tensor(np.array(int(sample[-1])))  # 将样本的最后一个元素（即标签）转换成一个整数，然后用 NumPy 将这个整数转换成一个 NumPy 数组，最后将这个 NumPy 数组转换成 PyTorch 张量并赋值给变量 label
        return img1, img2, kin_class, label

class TripletTrain(Dataset):
    def __init__(self,
                 number_triplet,
                 triplet_batch_size,  # 每个 batch 中 triplet 的数量
                 num_human_identities_per_batch=16, # 每个 batch 中不同的人数
                 # sample_path='../data/FIW/pairs/val_choose.txt',
                 # dataset_root="../data/FIW",
                 train_root="Train/train-faces",
                 dataset_root="../data/FIW",
                 images_size=112,
                 backbone='vits',
                 rand_mirror=True
                 ):
        # 将构造函数的参数保存为对象属性
        self.backbone=backbone
        self.dataset_root=dataset_root

        self.number_triplet=number_triplet
        self.triplet_batch_size=triplet_batch_size
        self.num_human_identities_per_batch=num_human_identities_per_batch

        self.images_size = images_size
        self.train_root=train_root

        self.triplets=self.generate_triplets()
        self.bias = 0
        self.rand_mirror = rand_mirror


    def load_family_and_person_dict(self):
        families_dict={}   # 定义了两个空字典，用于存储每个家庭及其成员的图像信息
        person_dict={}

        for f in os.listdir(os.path.join(self.dataset_root,self.train_root)):
            f_path=os.path.join(self.train_root,f) # 获取当前文件夹的路径
            families_dict[f_path]=[]  # 将当前家庭文件夹的路径作为键，对应的值设置为空列表
            for p in os.listdir(os.path.join(self.dataset_root,f_path)): # 遍历当前家庭文件夹下的所有文件夹（即人物文件夹）
                if p.startswith('MID'):  # 判断当前文件夹的名称是否以 "MID" 开头，以过滤掉不是人物文件夹的文件夹
                    p_path=os.path.join(f_path,p) # 获取当前人物文件夹的路径
                    families_dict[f_path].append(p_path) # 将当前人物文件夹的路径存储在当前家庭文件夹的值（即列表）中
                    person_dict[p_path]=[]
                    for img in os.listdir(os.path.join(self.dataset_root,p_path)):
                        person_dict[p_path].append(os.path.join(p_path,img)) # 将当前图像文件的路径存储在当前人物文件夹的值（即列表）中
        return families_dict,person_dict


    def generate_triplets(self):
        triplets = []
        self.families_dict, self.person_dict = self.load_family_and_person_dict()

        classes=list(self.families_dict.keys())  # 获取所有的人脸数据

        num_training_iterations_per_process = self.number_triplet / self.triplet_batch_size   # 每个进程需要生成的三元组数量
        progress_bar = int(num_training_iterations_per_process)    # 计算进度条

        for training_iteration in range(progress_bar):      # 按照生成的三元组数量进行迭代

            """
            For each batch: 
                - Randomly choose set amount of human identities (classes) for each batch

                  - For triplet in batch:
                      - Randomly choose anchor, positive and negative images for triplet loss
                      - Anchor and positive images in pos_class
                      - Negative image in neg_class
                      - At least, two images needed for anchor and positive images in pos_class
                      - Negative image should have different class as anchor and positive images by definition
            """

            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)   # 在数据集中随机选择num_human_identities_per_batch个类别

            for triplet in range(self.triplet_batch_size):  # 按照指定的三元组数量进行迭代

                pos_class = np.random.choice(classes_per_batch)  # 从当前batch的身份(classes_per_batch)中随机选择一个作为anchor和positive图像的类别
                neg_class = np.random.choice(classes_per_batch)

                while len(self.families_dict[pos_class]) < 2:  # 如果当前anchor和positive图像类别对应的数据数量小于2，则需要重新选择一个类别
                    pos_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)


                if len(self.families_dict[pos_class]) == 2:  # 如果anchor和positive图像类别对应的数据数量等于2，则可以直接随机选择anchor和positive图像
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                else:  # 否则，随机选择anchor和positive图像对应的数据
                    ianc = np.random.randint(0, len(self.families_dict[pos_class]))
                    ipos = np.random.randint(0, len(self.families_dict[pos_class]))

                    while ianc == ipos: # 如果选择的anchor和positive图像相同，则需要重新选择positive图像
                        ipos = np.random.randint(0, len(self.families_dict[pos_class]))


                ineg = np.random.randint(0, len(self.families_dict[neg_class]))  # 随机选择一个negative图像

                p_anc=self.families_dict[pos_class][ianc] # 获取正类别中的锚点人物
                p_pos = self.families_dict[pos_class][ipos] # 获取正类别中的正例人物
                p_neg = self.families_dict[neg_class][ineg]

                img_anc=random.choice(self.person_dict[p_anc]) # 随机选取锚点人物的一张照片
                img_pos = random.choice(self.person_dict[p_pos])
                img_neg = random.choice(self.person_dict[p_neg])

                triplets.append(
                    [
                        img_anc, # 锚点人物照片
                        img_pos, # 正例人物照片
                        img_neg,
                        pos_class, # 正例人物所在的类别
                        neg_class, # 负例人物所在的类别
                    ]
                )
        return triplets


    def __len__(self):
        return len(self.triplets)


    def set_bias(self,bias):
        self.bias=bias

    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone == 'vits':
            img = preprocess_input(img, version=2)

        if self.rand_mirror:
            # print("image randon flip")
            _rd = random.randint(0, 1)
            if _rd == 1:
                img = mx.nd.array(img)
                img = mx.ndarray.flip(data=img, axis=1)

        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item): # 用于读取数据集中的三元组（anchor, positive, negative）并返回其中的三个图像（anchor, positive, negative）
        sample = self.triplets[item+self.bias]
        anc_img,pos_img,neg_img=self.read_image(sample[0]),self.read_image(sample[1]),self.read_image(sample[2])
        return anc_img,pos_img,neg_img

