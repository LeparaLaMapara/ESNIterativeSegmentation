import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
from Attention import LSTMAttentionBlock

"""
Implementation of CNN+RNN/LSTM/GRU.
"""
class CRNN(nn.Module):
    def __init__(self, in_channels=3, sample_size=256, sample_duration=16, num_classes=100,
                hidden_size=512, num_layers=1, rnn_unit='LSTM'):
        super(CRNN, self).__init__()
        self.in_channels=in_channels
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.rnn_unit=rnn_unit

        # network params
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 128, 256, 512
        self.k1, self.k2, self.k3, self.k4 = (7, 7), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (1, 1), (1, 1), (1, 1)
        self.p1, self.p2, self.p3, self.p4 = (0, 0), (0, 0), (0, 0), (0, 0)
        self.d1, self.d2, self.d3, self.d4 = (1, 1), (1, 1), (1, 1), (1, 1)
        self.input_size = self.ch4
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # network architecture
        # in_channels=3 for rgb
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1, dilation=self.d1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2, dilation=self.d2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch2, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3, dilation=self.d3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.p4, dilation=self.d4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        if self.rnn_unit=='LSTM':
            self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout= 0.5 if self.num_layers > 1 else 0,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.rnn_unit=='GRU':
            self.lstm = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout= 0.5 if self.num_layers > 1 else 0,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.rnn_unit=='RNN':
            self.lstm = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout= 0.5 if self.num_layers > 1 else 0,
            num_layers=self.num_layers,
            batch_first=True,
        )

      
        self.fc1 = nn.Linear(self.hidden_size, self.num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # print(x.shape)
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # Conv
            out = self.conv1(x[:, :, t, :, :])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        hidden=None
        # use faster code paths
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(cnn_embed_seq, hidden)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = self.fc1(out[:, -1, :])
        out = self.act(out)

        return out


"""
Implementation of Resnet+LSTM
"""
class ResCRNN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                hidden_size=512, num_layers=1, rnn_unit="LSTM", 
                arch="resnet18",
                attention=False):
        super(ResCRNN, self).__init__()
        self.in_channels=in_channels
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.rnn_unit=rnn_unit

        # network params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
       
        if self.rnn_unit=='LSTM':
            self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.rnn_unit=='GRU':
            self.lstm = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.rnn_unit=='RNN':
            self.lstm = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.attention:
            self.attn_block = LSTMAttentionBlock(hidden_size=self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        hidden=None
        # use faster code paths
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(cnn_embed_seq, hidden)
        # MLP
        if self.attention:
            out = self.fc1(self.attn_block(out))
        else:
            # out: (batch, seq, feature), choose the last time step
            out = self.fc1(out[:, -1, :])

        return out



from torchesn.nn import ESN 



"""
Implementation of CNN+ESN.
"""
class CESN(nn.Module):
    def __init__(self, in_channels=3, sample_size=256, sample_duration=16, num_classes=100,
                hidden_size=512, num_layers=1, leaking_rate=0.05,spectral_radius=0.9, sparsity=0.2):
        super(CESN, self).__init__()
        self.in_channels=in_channels
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 128, 256, 512
        self.k1, self.k2, self.k3, self.k4 = (7, 7), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (1, 1), (1, 1), (1, 1)
        self.p1, self.p2, self.p3, self.p4 = (0, 0), (0, 0), (0, 0), (0, 0)
        self.d1, self.d2, self.d3, self.d4 = (1, 1), (1, 1), (1, 1), (1, 1)
        self.input_size = self.ch4
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaking_rate=   leaking_rate   
        self.spectral_radius=spectral_radius    
        self.sparsity  =sparsity                             

        # network architecture
        # in_channels=3 for rgb
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1, dilation=self.d1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2, dilation=self.d2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch2, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3, dilation=self.d3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.p4, dilation=self.d4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )

   
        self.lstm  =  ESN( input_size=self.input_size, 
                                        hidden_size=self.hidden_size,
                                        output_size=self.num_classes, 
                                        num_layers=self.num_layers,
                                        leaking_rate=self.leaking_rate, 
                                        spectral_radius=self.spectral_radius,
                                        density=self.sparsity,
                                        output_steps='last', 
                                        readout_training='inv',
                                        batch_first=True).cuda()

      
        # self.fc1 = nn.Linear(self.hidden_size, self.num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # print(x.shape)
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # Conv
            out = self.conv1(x[:, :, t, :, :])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        hidden=None
        # use faster code paths
        self.lstm.flatten_parameters()
        washouts = [int(0 * cnn_embed_seq.size(2))] *cnn_embed_seq.size(0)

        out, hidden = self.lstm(cnn_embed_seq, washouts)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = self.fc1(out[:, -1, :])
        out = self.act(out)

        return out


"""
Implementation of Resnet+ESN
"""
class ResCESN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                hidden_size=512, num_layers=1, 
                arch="resnet18",
                attention=False):
        super(ResCRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
       
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        if self.attention:
            self.attn_block = LSTMAttentionBlock(hidden_size=self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        hidden=None
        # use faster code paths
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(cnn_embed_seq, hidden)
        # MLP
        if self.attention:
            out = self.fc1(self.attn_block(out))
        else:
            # out: (batch, seq, feature), choose the last time step
            out = self.fc1(out[:, -1, :])

        return out
