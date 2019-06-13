# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:19:26 2019

@author: WT
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

##### LSTM #####
class lstm(nn.Module):
    def __init__(self, input_size, batch_size, lstm_hidden_size, num_layers, cuda1,\
                 bin_length=30):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.cuda1 = cuda1
        self.num_layers = num_layers
        self.bin_length = bin_length
        #self.hidden_state = self.init_hidden_lstm()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size,\
                             num_layers=num_layers, dropout=0.3, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size*self.bin_length, lstm_hidden_size*5)
        self.fc2 = nn.Linear(lstm_hidden_size*5, 2)
        
    def init_hidden_lstm(self, batch_size):
        grad = False
        if self.cuda1:
            return Variable(torch.randn(self.num_layers, batch_size, self.lstm_hidden_size),\
                        requires_grad=grad).cuda(),\
                Variable(torch.randn(self.num_layers, batch_size, self.lstm_hidden_size),\
                        requires_grad=grad).cuda()
        else:
            return Variable(torch.randn(self.num_layers, batch_size, self.lstm_hidden_size),\
                            requires_grad=grad),\
                    Variable(torch.randn(self.num_layers, batch_size, self.lstm_hidden_size),\
                            requires_grad=grad)
    
    def forward(self, seq):
        seq, _ = self.lstm1(seq, self.init_hidden_lstm(seq.shape[0]))
        seq = seq[:,:,-self.lstm_hidden_size:].reshape(seq.shape[0], -1)
        seq = torch.relu(self.fc1(seq))
        seq = torch.relu(self.fc2(seq))
        return seq

##### DenseNet #####
class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, droprate=0.1):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out,\
                               kernel_size=kernel_size, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm1d(c_out)
        self.droprate = droprate
        if self.droprate > 0:
            self.drop = nn.Dropout(p=self.droprate)

    def forward(self, s):
        out = torch.relu(self.batch_norm(self.conv(s)))
        if self.droprate > 0:
            out = self.drop(out)
        out = torch.cat((s, out), 1)
        return out
    
class BottleneckLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, droprate=0.1):
        super(BottleneckLayer, self).__init__()
        self.conv1 = nn.Conv1d(c_in, 4*c_out, kernel_size=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(4*c_out)
        self.droprate = droprate
        if self.droprate > 0:
            self.drop1 = nn.Dropout(p=self.droprate)
        self.conv2 = nn.Conv1d(4*c_out, c_out, kernel_size=kernel_size, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(c_out)

    def forward(self, x):
        out = torch.relu(self.batch_norm1(self.conv1(x)))
        if self.droprate > 0:
            out = self.drop1(out)
        out = torch.relu(self.batch_norm2(self.conv2(out)))
        out = torch.cat((x, out), 1)
        return out

class DownsampleLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownsampleLayer, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=1, stride=1,\
                               bias=False)
        self.batch_norm = nn.BatchNorm1d(c_out)
        self.avgpool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        out = torch.relu(self.batch_norm(self.conv1(x)))
        out = self.avgpool(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, c_in, k, num_layers, layertype=DenseLayer, kernel_size=3, droprate=0.1):
        super(DenseBlock, self).__init__()
        self.c_in = c_in
        self.k = k
        self.num_layers = num_layers
        for block in range(self.num_layers):
            setattr(self, "dense_%i" % block, layertype(self.c_in + block*self.k , self.k, \
                                                        droprate=droprate, kernel_size=kernel_size))
        self.last = self.c_in + (block + 1)*self.k
    
    def forward(self, seq):
        for block in range(self.num_layers):
            seq = getattr(self, "dense_%i" % block)(seq)
        return seq
    
class DenseNet(nn.Module):
    def __init__(self, c_in, c_out, batch_size):
        super(DenseNet, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.batch_size = batch_size
        self.k = 12
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels=self.c_in, out_channels=self.c_out,\
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(self.c_out)
        self.drop1 = nn.Dropout(p=0.01)
        # 1st dense + downsample block
        self.dense1 = DenseBlock(self.c_out, k=self.k, num_layers=15, layertype=DenseLayer,\
                                 droprate=0.1)
        self.ds1 = DownsampleLayer(self.dense1.last, int(self.dense1.last/2))
        # 2nd dense + downsample block
        self.dense2 = DenseBlock(int(self.dense1.last/2), k=self.k, num_layers=12, layertype=BottleneckLayer,\
                                 droprate=0.01)
        self.ds2 = DownsampleLayer(self.dense2.last, int(self.dense2.last/2))
        # 3rd dense + downsample block
        self.dense3 = DenseBlock(int(self.dense2.last/2), k=self.k, num_layers=9, layertype=BottleneckLayer,\
                                 droprate=0.01)
        self.ds3 = DownsampleLayer(self.dense3.last, int(self.dense3.last/2))
        # 4th dense + downsample block
        self.dense4 = DenseBlock(int(self.dense3.last/2), k=self.k, num_layers=5, layertype=BottleneckLayer,\
                                 droprate=0.1)
        self.ds4 = DownsampleLayer(self.dense4.last, int(self.dense4.last/2))
        # 5th dense + downsample block
        self.dense5 = DenseBlock(int(self.dense4.last/2), k=self.k, num_layers=3, layertype=BottleneckLayer,\
                                 droprate=0.01)
        self.ds5 = DownsampleLayer(self.dense5.last, int(self.dense5.last/2))
        # 6th dense + downsample block
        self.dense6 = DenseBlock(int(self.dense5.last/2), k=self.k, num_layers=2, layertype=BottleneckLayer,\
                                 droprate=0.01)
        self.ds6 = DownsampleLayer(self.dense6.last, int(self.dense6.last/2))
        # Classifier
        self.fc1 = nn.Linear(int(self.dense6.last/2)*14, 2)
        
    def forward(self, seq):
        # seq input = batch_size X seq_len X num_features
        seq = seq.permute(0,2,1); #print(seq.shape)
        seq = torch.relu(self.batch_norm1(self.conv1(seq))); #print(seq.shape)
        seq = self.drop1(seq); #print(seq.shape)
        seq = self.ds1(self.dense1(seq)); #print(seq.shape)
        seq = self.ds2(self.dense2(seq)); #print(seq.shape)
        seq = self.ds3(self.dense3(seq)); #print(seq.shape)
        seq = self.ds4(self.dense4(seq)); #print(seq.shape)
        seq = self.ds5(self.dense5(seq)); #print(seq.shape)
        seq = self.ds6(self.dense6(seq)); #print(seq.shape)
        seq = seq.reshape(len(seq[:,0,0]), -1); #print(seq.shape)
        seq = self.fc1(seq)
        return seq
    
class DenseNet_Block(nn.Module):
    def __init__(self, c_in, c_out, batch_size):
        super(DenseNet_Block, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.batch_size = batch_size
        self.k = 16
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels=self.c_in, out_channels=self.c_out,\
                               kernel_size=5, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(self.c_out)
        self.drop1 = nn.Dropout(p=0.025)
        # 1st dense + downsample block
        self.dense1 = DenseBlock(self.c_out, k=self.k, num_layers=4, kernel_size=3,\
                                 layertype=DenseLayer, droprate=0.05)
        self.ds1 = DownsampleLayer(self.dense1.last, int(self.dense1.last/2))
        # 2nd dense + downsample block
        self.dense2 = DenseBlock(int(self.dense1.last/2), k=self.k, num_layers=3, kernel_size=3,\
                                 layertype=BottleneckLayer, droprate=0.02)
        self.ds2 = DownsampleLayer(self.dense2.last, int(self.dense2.last/2))
        # 3rd dense + downsample block
        self.dense3 = DenseBlock(int(self.dense2.last/2), k=self.k, num_layers=3, layertype=BottleneckLayer,\
                                 droprate=0.03)
        self.ds3 = DownsampleLayer(self.dense3.last, int(self.dense3.last/2))
        # 4th dense + downsample block
        self.dense4 = DenseBlock(int(self.dense3.last/2), k=self.k, num_layers=3, layertype=BottleneckLayer,\
                                 droprate=0.03)
        self.ds4 = DownsampleLayer(self.dense4.last, int(self.dense4.last/2))
        # 5th dense + downsample block
        self.dense5 = DenseBlock(int(self.dense4.last/2), k=self.k, num_layers=2, layertype=BottleneckLayer,\
                                 droprate=0.03)
        self.ds5 = DownsampleLayer(self.dense5.last, int(self.dense5.last/2))
        # 6th dense + downsample block
        self.dense6 = DenseBlock(int(self.dense5.last/2), k=self.k, num_layers=2, layertype=BottleneckLayer,\
                                 droprate=0.03)
        self.ds6 = DownsampleLayer(self.dense6.last, int(self.dense6.last/2))
        
    def forward(self, seq):
        # seq input = batch_size X seq_len X num_features
        seq = seq.permute(0,2,1); #print(seq.shape)
        seq = torch.relu(self.batch_norm1(self.conv1(seq))); #print(seq.shape)
        seq = self.drop1(seq); #print(seq.shape)
        seq = self.ds1(self.dense1(seq)); #print(seq.shape)
        seq = self.ds2(self.dense2(seq)); #print(seq.shape)
        seq = self.ds3(self.dense3(seq)); #print(seq.shape)
        seq = self.ds4(self.dense4(seq)); #print(seq.shape)
        seq = self.ds5(self.dense5(seq)); #print(seq.shape)
        seq = self.ds6(self.dense6(seq)); #print(seq.shape)
        seq = torch.mean(seq, dim=2); #print(seq.shape)
        return seq

class DenseNetV2(nn.Module):
    def __init__(self, features_size, c_in, c_out, batch_size):
        super(DenseNetV2, self).__init__()
        self.features_size = features_size
        self.batch_size = batch_size
        for block in range(self.features_size):
            setattr(self, "densenet_%i" % block, DenseNet_Block(c_in=c_in, c_out=c_out,\
                                                                batch_size=self.batch_size))
        self.fc1 = nn.Linear(35*self.features_size, 2)
    
    def forward(self, seq):
        X = []
        for block in range(self.features_size):
            a = getattr(self, "densenet_%i" % block)(seq[:,:,block].unsqueeze(dim=2)); #print(a.shape)
            X.append(a)
        X = torch.cat([b for b in X], dim=1); #print(X.shape)
        X = self.fc1(X)
        return X
        

##### 1D CNN #####
class conv_net(nn.Module):
    def __init__(self, batch_size, cuda1, c_in, c_out, k_size=3, stride=1):
        super(conv_net, self).__init__()
        self.batch_size = batch_size
        self.cuda1 = cuda1
        self.c_in = c_in
        self.c_out = c_out
        self.k_size = k_size
        self.stride = stride
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv1 = nn.Conv1d(in_channels=self.c_in, out_channels=self.c_out,\
                               kernel_size=self.k_size, stride=self.stride)
        self.batch_norm1 = nn.BatchNorm1d(self.c_out)
        self.drop1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv1d(64, 64,\
                               kernel_size=self.k_size, stride=self.stride)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(p=0.3)
        
        self.conv3 = nn.Conv1d(64, 128,\
                               kernel_size=self.k_size, stride=self.stride)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(p=0.4)
        
        self.conv4 = nn.Conv1d(128, 256,\
                               kernel_size=self.k_size, stride=self.stride)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(p=0.3)
        
        self.conv5 = nn.Conv1d(256, 256,\
                               kernel_size=self.k_size, stride=self.stride)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        # F.max_pool1d(kernel_size=2)
    def forward(self, seq):
        # seq input = batch_size X seq_len X num_features
        seq = seq.transpose(0,2,1)
        seq = torch.relu(self.batch_norm1(self.conv1(seq)))
        seq = self.drop1(seq)
        
        seq = torch.relu(self.batch_norm2(self.conv2(seq)))
        seq = self.pool(seq)
        seq = self.drop2(seq)
        
        seq = torch.relu(self.batch_norm3(self.conv3(seq)))
        seq = self.drop3(seq)
        
        seq = torch.relu(self.batch_norm4(self.conv4(seq)))
        seq = self.drop4(seq)
        
        seq = torch.relu(self.batch_norm5(self.conv5(seq)))
        seq = self.drop5(seq)
        
        seq = torch.relu(self.fc1(seq))
        seq = self.fc2(seq)
        return seq

##### ResNet #####
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv1d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)
        s = torch.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if self.downsample:
            self.avgpool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            out = self.avgpool(out)
        out += residual
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,):
        super(ResNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(20):
            setattr(self, "res_%i" % block, ResBlock())
        self.fc1 = nn.Linear(128, 2)
    
    def forward(self, seq):
        # seq input = batch_size X seq_len X num_features
        seq = seq.permute(0,2,1); #print(seq.shape)
        seq = self.conv(seq)
        for block in range(20):
            seq = getattr(self, "res_%i" % block)(seq)
        seq = self.fc1(seq)
        return seq

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)