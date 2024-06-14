import torch.nn as nn
import torch
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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Framework(nn.Module):
    def __init__(self,I3D_classes, num_class=60, num_point=17, num_person=1, graph=None, graph_args=dict(),
                 drop_out=0, adaptive=True,is_train=True,mode='fusion',size=64):
        super(Framework,self).__init__()

        self.size=size
        self.count=0

        self.I3D = I3D_backbone(I3D_class=I3D_classes)
        # self.load_rgb_pretrain('/share/dingyuning/S2MAF/AQA7/type2/runs-59-1770.pt')
        self.load_rgb_pretrain('/share/dingyuning/S2MAF/AQA7/NN_sk2/runs-10-2000.pt')
        self.I3D.eval()
        # self.load_rgb_pretrain('AQA7_file/I3D.pt')
        # self.load_rgb_pretrain('/share/dingyuning/S2MAF/AQA7/NN_sk2/runs-6-1200.pt')


        # input()
        self.SkeletonModel = skeletonModel(num_class=num_class, num_point=num_point, num_person=num_person, graph=graph,
                                           graph_args=graph_args, in_channels=3,
                                           drop_out=drop_out, adaptive=adaptive)
        # self.load_ctr_pretrain('/share/dingyuning/S2MAF/AQA7/sk1/runs-113-791.pt')
        self.load_ctr_pretrain(weights='/share/dingyuning/S2MAF/AQA7/skFinal/runs-136-6800.pt')
        # self.load_ctr_pretrain(weights='/share/dingyuning/S2MAF/AQA7/fineAQA/runs-241-12050.pt')
        # self.load_ctr_pretrain(weights='/home/dingyuning/torch_demo/weights/AQA7_103/runs-542-27100.pt')
        # self.load_ctr_pretrain(weights='weights/AQA7_103/sk_finetune.pt')

        # self.load_ctr_pretrain(weights='weights/AQA7_103/sk_no_finetune.pt')
        # self.load_ctr_pretrain(weights='/share/dingyuning/S2MAF/AQA7/pre/pretrain.pt')
        # input()
        self.SkeletonModel.eval()
        self.l1 = torch.nn.Linear(in_features=1024, out_features=1)
        # self.l2 = torch.nn.Linear(in_features=256, out_features=1)
        self.l2 =  MLP(input_size=256,hidden_size=64,num_classes=1)
        self.l3 = torch.nn.Linear(in_features=1024, out_features=64)
        self.l4=torch.nn.Linear(in_features=1024+256, out_features=1)
        self.MLP = torch.nn.Linear(in_features=64 + 256, out_features=2)
        self.is_train=is_train
        self.dp=torch.nn.Dropout(0.5)
        self.range=100

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
                # input()
                diff2 = list(set(weights.keys()).difference(set(state.keys())))
                print('These weights is not in model:')
                for d in diff2:
                    print('  ' + d)
                state.update(weights)
                self.SkeletonModel.load_state_dict(state,strict=False)
    def load_rgb_pretrain(self,weights):
        if weights:
            if '.pkl' in weights:
                with open(weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(weights)
            for k, v in weights.items():
                print(k)
            weights = OrderedDict([[k.split('I3D.')[-1], v] for k, v in weights.items()])
            # print(keys)
            try:
                self.I3D.load_state_dict(weights)
            except:
                state = self.I3D.state_dict()
                # print('state keys are:')
                # print(state.keys())
                # print('weight keys are:')
                # print(weights.keys())
                # state = {k: v for k, v in self.SkeletonModel.state_dict() if k in weights.keys()}
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                input()
                diff2 = list(set(weights.keys()).difference(set(state.keys())))
                print('These weights is not in model:')
                for d in diff2:
                    print('  ' + d)
                input()
                state.update(weights)
                self.I3D.load_state_dict(state,strict=False)
    def forward(self,data,frame,label,mode='train'):
        # print(data.shape)
        # input()
        # self.count += 1
        # print(data.shape)
        with torch.no_grad():
            rgb_feature,rgb_sc = self.I3D(frame)
            skeleton_feature,sk_sc  = self.SkeletonModel(data)
        # print(rgb_sc)
        # rgb_sc = self.l1(rgb_feature)

        # fusion_sc = self.l4(torch.cat([skeleton_feature, rgb_feature], dim=1))
        rgb_feature = self.l3(rgb_feature)
        # rgb_feature=self.dp(rgb_feature)
        # sk_sc = self.l2(skeleton_feature)
        # print(skeleton_feature.shape,rgb_feature.shape)
        full_feature = torch.cat([skeleton_feature, rgb_feature], dim=1)

        selection = self.MLP(full_feature)
        # fusion_sc=torch.nn.Dropout()(fusion_sc)
        return rgb_sc, sk_sc, selection,label


if __name__=='__main__':
    model=Framework(I3D_classes=512,mode='skeleton')
    x=torch.randn(8,3,64,17,1)
    y=torch.randn(8,64,3,224,224)
    model((x,y))





