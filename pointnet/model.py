from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
#from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN

def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, num_points=2048, input_channgels=3):
        super(PointNetfeat, self).__init__()
        #self.stn = STN3d()
        self.feats = nn.Sequential(torch.nn.Conv1d(input_channgels, 64, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(True),
                                   torch.nn.Conv1d(64, 128, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(True),
                                   torch.nn.Conv1d(128, 1024, 1),
                                   nn.BatchNorm1d(1024),
                                   nn.MaxPool1d(num_points)
                                   )

        #self.conv1 = torch.nn.Conv1d(3, 64, 1)
        #self.bn1 = nn.BatchNorm1d(64)
        #self.relu1 = nn.ReLU(True)

        #self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.relu2 = nn.ReLU(True)

        #self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.bn3 = nn.BatchNorm1d(1024)
        #self.relu3 = nn.ReLU(True)


        #self.global_feat = global_feat
        #self.feature_transform = feature_transform
        #if self.feature_transform:
         #   self.fstn = STNkd(k=64)

    def forward(self, x):
       # n_pts = x.size()[2]
        #trans = self.stn(x)
        #x = x.transpose(2, 1)
       # x = torch.bmm(x, trans)
        #x = x.transpose(2, 1)
       # x = self.relu1(self.bn1(self.conv1(x)))

       # if self.feature_transform:
       #     trans_feat = self.fstn(x)
        #    x = x.transpose(2,1)
        #    x = torch.bmm(x, trans_feat)
        #    x = x.transpose(2,1)
        #else:
        #    trans_feat = None

       # pointfeat = x
        #x = self.relu2(self.bn2(self.conv2(x)))
        #x = self.bn3(self.conv3(x))
        x =self.feats(x)
        #x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
       # if self.global_feat:
        return x#, trans, trans_feat
       # else:
        #    x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        #    return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False, num_points=2048, input_channgels=3):
        super(PointNetCls, self).__init__()
        #self.feature_transform = feature_transform
        self.features = PointNetfeat(global_feat=True, feature_transform=feature_transform,num_points=num_points, input_channgels=input_channgels)
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, k)
        #self.dropout = nn.Dropout(p=0.3)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        #self.relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, k),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x) #, trans, trans_feat
       # x = F.relu(self.bn1(self.fc1(x)))
       # x = F.relu(self.bn2(self.dropout(self.fc2(x))))
       # x = self.fc3(x)
        x = self.classifier(x)
        return x#, trans, trans_feat
     #   return F.log_softmax(x, dim=1) , trans, trans_feat# TODO is it part of classifier?


class SAmodfeat(torch.nn.Module):
    def __init__(self):
        super(SAmodfeat, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return x


class PointNet2feat(torch.nn.Module):
    def __init__(self, global_feat = True, feature_transform = False,num_points=2048):
        super(PointNet2feat, self).__init__()

        self.feats = SAmodfeat()

    def forward(self, data):
        x = self.feats(data)
        #x = x.view(-1, 1024)
        return x

class PointNet2Cls(torch.nn.Module):
    def __init__(self, k=2, feature_transform=False,num_points=2048):
        super(PointNet2Cls, self).__init__()
        self.features = PointNet2feat(global_feat=True, feature_transform=feature_transform,num_points=num_points)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
