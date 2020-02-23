import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from dataset.autoaugment import ImageNetPolicy
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

class robot_pseudo_DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1280, 960), crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train', autoaug=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
         
        #https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        '''
        0 sky;  1 person; 2 two-wheel; 3 automobile; 4 sign
        5 light  6 building  7 sidewalk 8 road 
        '''
        self.id_to_trainid = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}

        for name in self.img_ids:
            img_file = osp.join(self.root, "%s/%s" % (self.set, name))
            if set == 'val':
                label_file = osp.join(self.root, "anno/%s" %name )
            else:
                label_file = osp.join(self.root, "pseudo_train_0.9/%s" %name )
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #tt = time.time()
        datafiles = self.files[index]
        name = datafiles["name"]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        if self.scale:
            random_scale = 0.8 + random.random()*0.4 # 0.8 - 1.2
            image = image.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.BICUBIC)
            label = label.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.NEAREST)
        else:
            image = image.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.BICUBIC)
            label = label.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.NEAREST)

        label = np.asarray(label, np.uint8)
            # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in list(self.id_to_trainid.items()):
            label_copy[label == k] = v
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)
        image = np.asarray(image, np.float32)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        if self.set == 'train':
            for i in range(10): #find hard samples
                x1 = random.randint(0, image.shape[1] - self.h)
                y1 = random.randint(0, image.shape[2] - self.w)
                tmp_label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]
                tmp_image = image[:, x1:x1+self.h, y1:y1+self.w]
                u =  np.unique(tmp_label_copy)
                if len(u) > 4:
                    break
                else:
                    print('RB: Too young too naive for %d times!'%i)
        else:        
            x1 = random.randint(0, image.shape[1] - self.h)
            y1 = random.randint(0, image.shape[2] - self.w)
            tmp_image = image[:, x1:x1+self.h, y1:y1+self.w]
            tmp_label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]

        image = tmp_image
        label_copy = tmp_label_copy

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label_copy = np.flip(label_copy, axis = 1)

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = robot_pseudo_DataSet('./data/Oxford_Robot_ICCV19', './dataset/robot_list/train.txt', mean=(0,0,0), set = 'train', autoaug=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, _, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.save('Robot_pseudo_Demo.jpg')
        break
