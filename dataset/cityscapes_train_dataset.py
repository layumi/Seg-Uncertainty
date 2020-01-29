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
from PIL import Image
from dataset.autoaugment import ImageNetPolicy
import time

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train', autoaug=False):
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
        project Cityscapes to Oxford Robot
        7 road -> 8;  8 sidewalk -> 7; building 11 -> 6; wall 12 -> 255;
        fence 13 -> 255; pole 17-> 255: light 19 -> 5; sign 20->4;
        vegetation -> 255; terrain -> 255; sky 23 -> 0; person 24 -> 1 ;
        rider 25 -> 1 ; car 26 -> 3; truck 27 ->3; bus 28 ->3; train 31->255;
        motorcycle 32->2 ; bike 33 -> 2;

        '''
        self.id_to_trainid = {7: 8, 8: 7, 11: 6, 
                              19: 5, 20: 4, 23: 0, 24: 1, 25: 1,
                              26: 3, 27: 3, 28: 3, 32: 2, 33: 2}

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name.replace('leftImg8bit', 'gtFine_labelIds') ))
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

        image, label = Image.open(datafiles["img"]).convert('RGB'), Image.open(datafiles["label"])
        # resize
        image, label = image.resize(self.resize_size, Image.BICUBIC), label.resize(self.resize_size, Image.NEAREST)
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)

        image, label = np.asarray(image, np.float32), np.asarray(label, np.uint8)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in list(self.id_to_trainid.items()):
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        x1 = random.randint(0, image.shape[1] - self.h)
        y1 = random.randint(0, image.shape[2] - self.w)
        image = image[:, x1:x1+self.h, y1:y1+self.w]
        label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label_copy = np.flip(label_copy, axis = 1)
        #print('Time used: {} sec'.format(time.time()-tt))
        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = cityscapesDataSet('./data/Cityscapes/data', './dataset/cityscapes_list/train.txt', mean=(0,0,0), set = 'train')
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, _, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.save('Cityscape_Demo.jpg')
        break
