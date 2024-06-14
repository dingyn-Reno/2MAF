import torch.nn as nn
import torch
from info_nce import*
try:
    from model.Backbone import I3D_backbone
    from model.ctrgcn import Model as skeletonModel
except:
    from Backbone import I3D_backbone
    from ctrgcn import Model as skeletonModel
import logging

import torchvision
import pickle
from collections import OrderedDict
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

import numpy as np

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Framework(nn.Module):
    def __init__(self,I3D_classes, num_class=60, num_point=17, num_person=1, graph=None, graph_args=dict(),
                 drop_out=0, adaptive=True,is_train=True,mode='fusion',size=64):
        super(Framework,self).__init__()
        self.size=size
        self.count=0
        self.mode=mode
        if mode=='skeleton':
            self.SkeletonModel = skeletonModel(num_class=num_class, num_point=num_point, num_person=num_person, graph=graph,
                                       graph_args=graph_args, in_channels=3,
                                       drop_out=drop_out, adaptive=adaptive)
            # self.load_ctr_pretrain(weights='/home/dingyuning/torch_demo/weights/skeleton_model_M3FS.pt')
            self.MLP = torch.nn.Linear(in_features=256, out_features=1)
            self.dropout=nn.Dropout(drop_out)
        elif mode=='RGB':
            self.I3D = I3D_backbone(I3D_class=I3D_classes)
            self.I3D.load_pretrain('/home/dingyuning/torch_demo/weights/final_2MAF_AQA.pth')
            self.I3D.eval()
            self.MLP = torch.nn.Linear(in_features=16384, out_features=1)


    def load_ctr_pretrain(self, weights):
        if weights:
            if '.pkl' in weights:
                with open(weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(weights)
            for k, v in weights.items():
                print(k)
            weights = OrderedDict([[k.split('SkeletonModel.')[-1], v] for k, v in weights.items()])
            # print(keys)
            try:
                self.SkeletonModel.load_state_dict(weights)
            except:
                state = self.SkeletonModel.state_dict()
                # print('state keys are:')
                # print(state.keys())
                # print('weight keys are:')
                # print(weights.keys())
                # state = {k: v for k, v in self.SkeletonModel.state_dict() if k in weights.keys()}
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.SkeletonModel.load_state_dict(state,strict=False)



    def loss(self,x1,x2,N):
        N,F=x1.shape
        loss = InfoNCE()
        # print(x1.shape,x2.shape)
        return loss(x1,x2)

    def variance_mask(self,data,SL):
        N, C, T, V, M = data.shape
        values, indices = torch.topk(SL, k=self.size, largest=True)
        values, indices = torch.sort(indices, dim=1, descending=False)
        data1 = torch.permute(data, (0, 2, 1, 3, 4))
        data1 = data1[:, values, :, :, :]
        data1 = torch.diagonal(data1[:, :, ])
        data1 = torch.permute(data1, (4, 1, 0, 2, 3))
        return data1

    def temporal_mask(self,data):
        N, C, T, V, M = data.shape
        gap = T // self.size
        start = T % self.size
        data2 = data[:, :, start::gap, :, :]
        return data2
    def forward(self,data,SL,label,mode='train'):
        # print(data.shape)
        if self.mode=='RGB':
            with torch.no_grad():
                full_feature=self.I3D(data)
            return self.MLP(full_feature),label
        elif self.mode=='skeleton':
            data1=self.variance_mask(data,SL)
            data2=self.temporal_mask(data)
            # data2=data[:,:,0:self.size,:,:]
            # self.count +=1
            # print(data.shape)
            full_feature1 = self.SkeletonModel(data1)[0]
            full_feature2= self.SkeletonModel(data2)[0]
            loss=self.loss(full_feature1,full_feature2,N)
            return self.MLP(full_feature1),loss


if __name__=='__main__':
    model=Framework(I3D_classes=512,mode='skeleton')
    x=torch.randn(8,3,103,17,1)
    label=torch.randn(8,1)
    a,b=model(x)
    print(a.shape)





