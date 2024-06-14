import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from decord import VideoReader
from decord import cpu, gpu
from decord import bridge
import torch.nn as nn
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer
import pickle
# operation
try:
    from . import tools
except:
    import tools
import scipy.io as IO

class VideoDataset(Dataset):
    def __init__(self, data_path,label_path,diff_path,split,random_choose=True,random_move=True,
                 window_size=256, clip_len=256, size=224, debug=None, mode='skeleton',centralization=True,choose_mode='temporal',mmap=True,actiontype=0):

        self.clip_len = clip_len
        self.size = size
        self.mode = mode
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.diff_path=diff_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.centralization = centralization
        self.choose_mode=choose_mode
        self.clip_length=clip_len
        self.load_data(mmap)
        self.actiontype = actiontype
        self.split=split
        # if self.mode=='rgb' or self.mode=='fusion':
        self.frame_names = self.load_frame()

    def load_frame(self):
          # 读取.mat文件
        if (self.split == 'train'):
            mat_data = IO.loadmat('/share/Diving/split_300_70/training_idx.mat')
            list_tmp = mat_data['training_idx'][0]
            data = []
            for tmp in list_tmp:
                data.append('/share/Diving/diving_samples_len_151_lstm/'+'{:03}.avi'.format(tmp)
                            )
            ret = data
            # print(ret)
        else:
            mat_data = IO.loadmat('/share/Diving/split_300_70/testing_idx.mat')
            list_tmp = mat_data['testing_idx'][0]
            data = []
            for tmp in list_tmp:
                data.append('/share/Diving/diving_samples_len_151_lstm/' + '{:03}.avi'.format(tmp)
                            )
            ret = data

        # print(ret)
        return ret
    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f)
        with open(self.diff_path, 'rb') as f:
            self.diff = pickle.load(f)
        self.sample_name=self.label
        # load data
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]

        # print(type(self.label),type(self.sample_name),type(self.data))
    def video_to_tensor(self, filename):
        try:
            # print('file name is {}'.format(filename))
            vr = VideoReader(filename, ctx=cpu(0))
            indexes=[i for i in range(0,len(vr))]
            x=bridge.to_torch(vr.get_batch(indexes))
        except:
            try:
                print('file not exist and searching again:{}'.format(filename))
                filename = filename.replace('/p', '/P')
                vr = VideoReader(filename, ctx=cpu(0))
                indexes = [i for i in range(0, len(vr))]
                x = bridge.to_torch(vr.get_batch(indexes))
            except:
                print('file not exist:{}'.format(filename))
                x = torch.zeros((self.clip_len, 3, self.size, self.size), dtype=torch.float32)
                return x
        n = x.shape[0]
        x=torch.permute(x,(0,3,1,2)).type(torch.float32)
        # print(x)
        if n >= self.clip_len:
            x = x[:self.clip_len]  # 取前 96 个元素
        else:
            zeros = torch.zeros((self.clip_len - n,) + x.shape[1:], dtype=x.dtype)  # 构造全零张量
            x = torch.cat((x, zeros), dim=0)  # 在第 0 维上拼接全零张量和 x

        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        # x /= 256
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # print(x.size())
        return x

    def __getitem__(self, index):
        if self.mode=='fusion':
            # get data
            data_numpy = np.array(self.data[index])
            # print(data_numpy.shape)

            data_numpy = tools.random_resize(data_numpy, self.clip_length)
            C, T, V, M = data_numpy.shape
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            data_numpy = data_numpy[:, :, :, :] - data_numpy[:, :, 8:9, :]

            # print(data_numpy.shape)
            label = self.label[index]
            ret = self.frame_names[index]
            # print(data_numpy.shape)
            video=self.video_to_tensor(ret)
            # exit(0)
            # processing
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            data_numpy = data_numpy[:, :, :, :] - data_numpy[:, :, 8:9, :]
            diff=self.diff[index]
            return data_numpy, video, label,diff, index

        elif self.mode=='skeleton':
            # get data
            data_numpy = np.array(self.data[index])
            # print(data_numpy.shape)

            data_numpy = tools.random_resize(data_numpy, self.clip_length)
            C, T, V, M = data_numpy.shape
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            data_numpy = data_numpy[:, :, :, :] - data_numpy[:, :, 8:9, :]

            # print(data_numpy.shape)
            label = self.label[index]
            # exit(0)
            # processing
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            data_numpy = data_numpy[:, :, :, :] - data_numpy[:, :, 8:9, :]
            diff=self.diff[index]
            return data_numpy,  label,diff, index

        elif self.mode=='rgb':
            ret = self.frame_names[index]
            # print(data_numpy.shape)
            label = self.label[index]

            video=self.video_to_tensor(ret)
            diff = self.diff[index]
            return video, label,diff, index
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = VideoDataset(label_path='/root/autodl-tmp/M3FS/skeleton/train_score.pkl',
                           name_path='/root/autodl-tmp/M3FS/skeleton/train_name.pkl',
                           data_path='/root/autodl-tmp/M3FS/skeleton/train_data.pkl',
                           video_dir='/root/autodl-tmp/M3FS/all_data/skating256_all/video/',
                            clip_len=64,
                           )
    x = dataset[0][1].shape
    print(x)