# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
from skimage import io
from skimage.transform import resize
import numpy as np
import random
import cfg
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image

class datagen_srnet(Dataset):
    def __init__(self, cfg, torp='train'):
        if (torp == 'train'):
            self.data_dir = cfg.data_dir
            self.t_t_dir = cfg.t_t_dir
            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
            self.name_list = os.listdir(os.path.join(self.data_dir, self.t_t_dir))
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        # print(img_name)

        # i_t = Image.open(os.path.join(cfg.data_dir, cfg.mask_t_dir, img_name)).convert("L")
        # i_t =   self.transform(i_t)
        # i_s = Image.open(os.path.join(cfg.data_dir, cfg.i_t_dir, img_name)).convert("L")
        # i_s = self.transform(i_s)
        # # as_gray mean single channel
        # t_sk = Image.open(os.path.join(cfg.data_dir, cfg.t_sk_dir, img_name)).convert("L")
        # t_sk = self.transform(t_sk)
        # t_t = Image.open(os.path.join(cfg.data_dir, cfg.t_t_dir, img_name)).convert("L")
        # t_t = self.transform(t_t)
        # print("i_t", i_t)
        i_t = Image.open(os.path.join(cfg.data_dir, cfg.mask_t_dir, img_name))
        i_t = self.transform(i_t)
        i_s = Image.open(os.path.join(cfg.data_dir, cfg.i_t_dir, img_name))
        i_s = self.transform(i_s)
        # as_gray mean single channel
        t_sk = Image.open(os.path.join(cfg.data_dir, cfg.t_sk_dir, img_name)).convert("L")
        t_sk = self.transform(t_sk)
        t_t = Image.open(os.path.join(cfg.data_dir, cfg.t_t_dir, img_name))
        t_t = self.transform(t_t)
        return [i_t, i_s, t_sk, t_t]


class example_dataset(Dataset):

    def __init__(self, data_dir=cfg.example_data_dir, transform=None):
        self.files = os.listdir(data_dir)
        self.files = [i.split('_')[0] + '_' for i in self.files]
        self.files = list(set(self.files))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        # i_t = Image.open(os.path.join(cfg.example_data_dir, img_name + 'i_t.png')).convert("L")
        # i_s = Image.open(os.path.join(cfg.example_data_dir, img_name + 'i_s.png')).convert("L")
        i_t = Image.open(os.path.join(cfg.example_data_dir, img_name + 'i_t.png'))
        i_s = Image.open(os.path.join(cfg.example_data_dir, img_name + 'i_s.png'))
        # h, w = i_t.shape[:2]
        # scale_ratio = cfg.data_shape[0] / h
        # to_h = cfg.data_shape[0]
        # to_w = int(round(int(w * scale_ratio) / 8)) * 8
        # to_scale = (to_h, to_w)
        #
        # i_t = resize(i_t, to_scale, preserve_range=True)
        # i_s = resize(i_s, to_scale, preserve_range=True)
        #
        i_t = self.transform(i_t)
        i_s = self.transform(i_s)
        # if self.transform:
        #     sample = self.transform(sample)

        return i_t,i_s,img_name


class To_tensor(object):
    def __call__(self, sample):
        i_t, i_s, img_name = sample

        i_t = i_t.transpose((2, 0, 1)) / 127.5 - 1
        i_s = i_s.transpose((2, 0, 1)) / 127.5 - 1

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)

        return (i_t.float(), i_s.float(), img_name)


