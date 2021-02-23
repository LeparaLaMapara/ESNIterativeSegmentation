import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

import numpy as np

from scipy import sparse






class ImageRNN(nn.Module):
    def __init__(self,  n_inputs:int=64, 
                        n_outputs:int=4096, 
                        n_hidden:int=256, 
                        n_layers:int=1, 
                        bidirectional:bool=False,
                        batch_first:bool=True)->torch.Tensor():
        super(ImageRNN, self).__init__()
        self.n_inputs      = n_inputs
        self.n_hidden      = n_hidden
        self.n_outputs     = n_outputs
        self.n_layers      = n_layers
        self.bidirectional = bidirectional
        self.batch_first   = batch_first
        self.rnn          = nn.RNN(input_size=self.n_inputs, 
                                    hidden_size=self.n_hidden, 
                                    num_layers=self.n_layers, 
                                    dropout= 0.5 if self.n_layers > 1 else 0,
                                    bidirectional=self.bidirectional,
                                    batch_first=self.batch_first )
        
        self.FC         = nn.Linear(self.n_hidden*2, self.n_outputs) if \
        (self.bidirectional) else nn.Linear(self.n_hidden, self.n_outputs)                       
        self.out        = nn.Sigmoid()
        
    def init_hidden(self, X, device=None): # input 4D tensor: (batch size, channels, width, height)
        # initialize the hidden and cell state to zero
        # vectors:(number of layer, batch size, number of hidden nodes)
        h0 = torch.zeros(2*self.n_layers,  X.size(0), self.n_hidden) if self.bidirectional else torch.zeros(self.n_layers,  X.size(0), self.n_hidden)

        if device is not None:
            h0 = h0.to(device)
        self.hidden = h0
            
    def forward(self, X): 
        #  x --> (batch size, sequence length, input)
        
        lstm_out, self.hidden = self.rnn(X, self.hidden) 
#         print(lstm_out.shape)
        out = self.FC(lstm_out[:, -1, :])
#         out = self.out(out)
        return out
    
    def predict(self, X, max_length=100):
        """Generate captions for given image features using greedy search."""
        outputs = []
        
        batch_size, num_channels, H, W = X.view(-1, 2, 32, 32).shape
        self.x_temp = X.view(-1, num_channels, H, W)[:,0][np.newaxis]
#         Seg   = torch.random((*X.shape))
        
#         print('X', X.shape)
        for i in np.arange(max_length):
            lstm_out, self.hidden = self.rnn(X, self.hidden)    # lstm_out: (batch_size, 1, hidden_size)
            out = self.FC(lstm_out[:, -1, :])                   # outputs:  (batch_size, output_size)
#             print(f'out {i}', out.shape)
#             out = self.out(out)
            outputs.append(out)
            
            X = torch.stack(outputs, dim=1)
#             print('X', X.view(-1, 1, H, W).shape)
#             self.x_temp = self.x_temp.repeat_interleave(X.view(-1, 1, H, W).shape[0], dim=0)
#             print('x', self.x_temp.shape)
#             print()
            
            X = torch.cat((self.x_temp , out.view(-1, 1, H, W) ), dim=2) # concat along 
            X = X.view(-1, H, W*num_channels)        # inputs: (batch_size, 1, embed_size)
#             print('testt', X.shape)
            
        prediction = torch.stack(outputs, 1)                    # sampled_ids: (batch_size, max_seq_length)
#         prediction = prediction
        prediction = self.out(prediction)
#         print('pred', prediction.shape)
        return prediction[:,-1,:]

    
class ImageGRU(nn.Module):
    def __init__(self,  n_inputs:int=49, 
                        n_outputs:int=4096,
                        n_hidden:int=256, 
                        n_layers:int=1, 
                        bidirectional:bool=False):

        """
        Takes a 1D flatten images.
        """
        super(ImageGRU, self).__init__()
        self.n_inputs   = n_inputs
        self.n_hidden   = n_hidden
        self.n_outputs  = n_outputs
        self.n_layers   = n_layers
        self.bidirectional = bidirectional
        # pass cnn extracted features to LSTM (i.e. learn the temporal information in the extracted features)
        self.gru       = nn.GRU(input_size=self.n_inputs, 
        hidden_size=self.n_hidden, 
        num_layers=self.n_layers, 
        bidirectional=self.bidirectional, batch_first=True)
        # grow the output signal from LSTM gradaully using the leaner layers
        # and normizile the weights using the batch norm
        if (self.bidirectional):
            self.FC         = nn.Sequential(
                                            nn.Linear(self.n_hidden*2, self.n_outputs),
                                            nn.Dropout(p=0.5),
                                            nn.Sigmoid()
                                                )

        else:
            self.FC         = nn.Sequential(
                                            nn.Linear(self.n_hidden, self.n_outputs),
                                            nn.Dropout(p=0.5),
                                            nn.Sigmoid()
                                            )

        
    def init_hidden(self, batch_size, device=None): # input 4D tensor: (batch size, channels, width, height)
        # initialize the hidden and cell state to zero
        # vectors:(number of layer, sequence length, number of hidden nodes)
        if (self.bidirectional):
            h0 = torch.zeros(2*self.n_layers, batch_size , self.n_hidden)
        else:
            h0 = torch.zeros(self.n_layers,  batch_size, self.n_hidden)

        if device is not None:
            h0 = h0.to(device)
        self.hidden = h0

        
    def forward(self, X): # X: tensor of shape (batch_size, channels, width, height)
        # batch_size, num_channel, W, H = X.shape
        # transforms X to dimensions: (batch_size, seq_length, input_size)
        # X = X.view(batch_size, H, num_channel*W)
        # forward propagate LSTM
        lstm_out, self.hidden = self.gru(X, self.hidden) # lstm_out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step 
        # grow the output from the LSTM 
        out = self.FC(lstm_out[:, -1, :])
        return out


class ImageLSTM(nn.Module):
    def __init__(self,  n_inputs:int=49, 
                        n_outputs:int=4096, 
                        n_hidden:int=256, 
                        n_layers:int=1, 
                        bidirectional:bool=False):
        """
        Takes a 1D flatten images.
        """
        super(ImageLSTM, self).__init__()
        self.n_inputs   = n_inputs
        self.n_hidden   = n_hidden
        self.n_outputs  = n_outputs
        self.n_layers   = n_layers
        self.bidirectional = bidirectional
        self.lstm       = nn.LSTM(  input_size=self.n_inputs, 
                                    hidden_size=self.n_hidden, 
                                    num_layers=self.n_layers, 
                                    dropout  = 0.5 if self.n_layers>1 else 0,
                                    bidirectional=self.bidirectional,
                                    batch_first=True)
        if (self.bidirectional):
            self.FC         = nn.Sequential(
                                            nn.Linear(self.n_hidden*2, self.n_outputs),
                                            nn.Dropout(p=0.5),
                                            nn.Sigmoid()
                                                )

        else:
            self.FC         = nn.Sequential(
                                            nn.Linear(self.n_hidden, self.n_outputs),
                                            # nn.Dropout(p=0.5),
                                            nn.Sigmoid()
                                            )

        
    def init_hidden(self, batch_size, device=None): # input 4D tensor: (batch size, channels, width, height)
        # initialize the hidden and cell state to zero
        # vectors:(number of layer, sequence length, number of hidden nodes)
        if (self.bidirectional):
            h0 = torch.zeros(2*self.n_layers,  batch_size, self.n_hidden)
            c0 = torch.zeros(2*self.n_layers,  batch_size, self.n_hidden)
        else:
            h0 = torch.zeros(self.n_layers,  batch_size, self.n_hidden)
            c0 = torch.zeros(self.n_layers,  batch_size, self.n_hidden)

        if device is not None:
            h0 = h0.to(device)
            c0 = c0.to(device)
        self.hidden = (h0,c0)

        
        
    def forward(self, X): # X: tensor of shape (batch_size, channels, width, height)
        # forward propagate LSTM
        lstm_out, self.hidden = self.lstm(X, self.hidden) # lstm_out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step 
        out = self.FC(lstm_out[:, -1, :])
        return out



# class ImageRNN(nn.Module):
#     def __init__(self,  n_inputs:int=49,
#                         n_outputs:int=4096, 
#                         n_hidden:int=512, 
#                         n_layers:int=1, 
#                         bidirectional:bool=False):
#         """
#         Takes a 1D flatten images.
#         """
#         super(ImageRNN, self).__init__()
#         self.n_inputs   = n_inputs
#         self.n_hidden   = n_hidden
#         self.n_outputs  = n_outputs
#         self.n_layers   = n_layers
#         self.bidirectional = bidirectional
#         # pass cnn extracted features to RNN (i.e. learn the temporal information in the extracted features)
#         self.rnn       = nn.RNN(self.n_inputs, 
#         self.n_hidden, 
#         self.n_layers, 
#         bidirectional=self.bidirectional, 
#         batch_first=True)
#         # grow the output signal from RNN gradaully using the leaner layers
#         # and normizile the weights using the batch norm
#         if (self.bidirectional):
#             self.FC         = nn.Sequential(
#                                             nn.Linear(self.n_hidden*2, self.n_outputs),
#                                             nn.Dropout(p=0.5),
#                                             nn.Sigmoid()
#                                                 )

#         else:
#             self.FC         = nn.Sequential(
#                                             nn.Linear(self.n_hidden, self.n_outputs),
#                                             nn.Dropout(p=0.5),
#                                             nn.Sigmoid()
#                                             )

        
#     def init_hidden(self, batch_size, device=None): # input 4D tensor: (batch size, channels, width, height)
#         # initialize the hidden and cell state to zero
#         # vectors:(number of layer, sequence length, number of hidden nodes)
#         if (self.bidirectional):
#             h0 = torch.zeros(2*self.n_layers, batch_size, self.n_hidden)
#         else:
#             h0 = torch.zeros(self.n_layers,   batch_size, self.n_hidden)
            
#         if device is not None:
#             h0 = h0.to(device)
#         self.hidden = h0

        
#     def forward(self, X): # X: tensor of shape (batch_size, channels, width, height)
#         # transforms X to dimensions: (batch_size, seq_length, input_size) 
#         # forward propagate RNN
#         rnn_out, self.hidden = self.rnn(X, self.hidden) # rnn_out: tensor of shape (batch_size, seq_length, hidden_size)
#         # Decode the hidden state of the last time step 
#         # grow the output from the LSTM 
#         out = self.FC(rnn_out[:, -1, :])
#         return out


class ESN(nn.Module):
    def __init__(self,  input_size:int, 
                        reservoir_size:int, 
                        output_size:int, 
                        online_training:bool=False, 
                        leaking_rate:float=0.9, 
                        sparsity:float=0.8, 
                        spectral_radius:float=0.9, 
                        input_scaling:float=1.0, 
                        beta:float=1e-6, 
                        seed:int=0):
        super(ESN, self).__init__()
        self.input_size     = input_size
        self.reservoir_size = reservoir_size
        self.output_size    = output_size
        self.leaking_rate   = leaking_rate
        self.sparsity       = sparsity
        self.spectral_radius= spectral_radius
        self.input_scaling  = input_scaling
        self.online_training= online_training
        self.beta           = beta
        self.seed           = seed
        self.reset_reservoir()
        
    def extra_repr(self):
        return '((input_matrix): Inputs(({} ,{})) \n (reservior_matrix): Reservoir(({} ,{}))'.format(
            self.W_in.shape[0], self.W_in.shape[1], self.W.shape[0], self.W.shape[1])
        
    def reset_reservoir(self):
        torch.manual_seed(self.seed)
        self.reset_reservoir_state()
        self.create_reservoir() 
        self.register_parameters()
        
    def create_reservoir(self):
        """"
        creates reservoir.
        """
        # initialize input-reservoir connection matrix as N X Nx
        self.W_in  =  torch.Tensor(self.reservoir_size, self.input_size ).uniform_(-1, 1)
        self.W_in  =  self.W_in * (self.input_scaling / torch.max(torch.abs(self.W_in)))
        self.bias  =  torch.ones(self.input_size + 1) * (self.input_scaling)
#         self.W_in[0,:] = self.bias

        # initialize reservoir-reservoir connection matrix as N X N
        self.W     = torch.Tensor(self.reservoir_size, self.reservoir_size).uniform_(-1, 1)
        self.W     = self.get_sparse_matrix(self.W)
        self.W     = self.W * (self.spectral_radius / self.get_spectral_radius(self.W))

        # readout-initialize reservoir-output connection matrix as Ny X ( Nx + N + Ny)
        self.readout = nn.Linear(self.reservoir_size, self.output_size, bias=True)
        
        # register as parameters
        self.W_in  = nn.Parameter(self.W_in, requires_grad=False)
        self.W     = nn.Parameter(self.W,    requires_grad=False)
        
        # if off-line training mode
        if not self.online_training: 
            print("TRAINING OFFLINE")
            self.readout.weight.requires_grad = False
            
    def register_parameters(self):
        self.register_parameter('inputs_matrix',    self.W_in)
        self.register_parameter('reservoir_matrix', self.W)
        
    def reset_reservoir_state(self):
        """"
        reset the reservoir state to zero.
        """
        self.x = torch.zeros((1, self.reservoir_size))
        self.register_buffer('states', self.x)

    def get_spectral_radius(self, W):
        """"
        computes spectral radius of matrix (i.e. maximum eigenvalue).
        """
        return torch.max(torch.abs(torch.eig(W)[0])).item()

    def get_sparse_matrix(self, w):
        """"
        generates sparse matrix.
        """
        row, col = w.shape
        w = w.view(int(row * col))
        # permute the indices for the matrix
        zero_weights_indices = torch.randperm(int(row * col)) 
        # select indices based on the sparsity
        zero_weights_indices = zero_weights_indices[:int(row * col * ( self.sparsity))] 
        # make those indics
        w[zero_weights_indices] = 0 
        w = w.view(row , col)
        return w

    def activation_function(self, s):
        """
        reservoir activation function.
        """
        a = torch.tanh(s)
        return a
    

    def update_state(self, u):
        """
        update the states of the reservoir.
        """        
        self.x = self.x.to(u.device)
        # print(u.device, self.W_in.device, self.x.device, self.W.device)
        a = self.activation_function(F.linear(u, self.W_in) + F.linear(self.x, self.W))
        self.x = (1-self.leaking_rate) * self.x + self.leaking_rate * a

        
    def reservoir_states(self, U):
        """
        computes reservoir states. 
        Transform input into a higher dimensional feature space.
        """
        N         = U.shape[0]
        X         = torch.zeros((N, self.reservoir_size)).to(U.device)
        for i, input_state in enumerate(U):
            self.update_state(input_state)
            X[i,:] = self.x[0,:]
        return X 
        
        
    def offline_readouts(self, X, Y):
        """
        computes readouts using closed form Moore-Penrose pseudo-inverse.
        """
#         print("Offline Mode")
        N = Y.shape[0]
        Ytarget = torch.zeros((N, self.output_size))
        Ytarget[torch.from_numpy(np.arange(N)).long(), Y.long()] = 1.0
        with torch.no_grad():
            XTX = torch.mm(X.T,X) + self.beta * torch.eye(self.reservoir_size).to(X.device)
            XTY = torch.mm(X.T,Ytarget.to(X.device))
            self.W_out = torch.mm(torch.pinverse(XTX), XTY).T
            
            self.readout.bias   = nn.Parameter(self.W_out[:, 0])
            self.readout.weight = nn.Parameter(self.W_out[:, 0:])
        outputs = self.readout(X) 
        return outputs
        
    def online_readouts(self, X):
        """
        computes readouts using iterative methods.
        """
#         print("Online Mode")
        with torch.enable_grad():
            outputs = self.readout(X) 
        return outputs
        
    def forward(self, U, Y=None):
        X = self.reservoir_states(U)   
        if (self.online_training) or (Y is None):
            outputs = self.online_readouts(X)          
        else:
            outputs = self.offline_readouts(X, Y)
        return  torch.sigmoid(outputs)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class:int=1):
        super().__init__()
                
        self.dconv_down1 = double_conv(2, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        out=  torch.sigmoid(out.reshape(out.size(0), -1))
        return out


class CNN(nn.Module):
    def __init__(self, n_outputs):
        super(Net, self).__init__()
        self.n_outputs = n_outputs
        self.conv1     = nn.Conv2d(4, 6, 5, bias = False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool      = nn.MaxPool2d(2, 2)
        self.conv2     = nn.Conv2d(6, 16, 5, bias = False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.fc1       = nn.Linear(16 * 5 * 5, 120, bias = False)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2       = nn.Linear(120, 84, bias = False)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3       = nn.Linear(84, self.n_outputs, bias = False)
        nn.init.xavier_uniform_(self.fc3.weight)

        
    def forward(self, x, training=False):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(F.relu(self.fc2(x)), training=training)
        x = torch.sigmoid(self.fc3(x))
        return x



class ESN:
    def __init__(self, d_input, d_reservoir, d_output, sparsity,
     spectral_radius, input_scaling, leaking_rate, seed_value=7):
        self.d_input         = d_input 
        self.d_reservoir     = d_reservoir
        self.d_auxilary      = d_reservoir
        self.d_output        = d_output
        self.sparsity        = sparsity
        self.spectral_radius = spectral_radius
        self.input_scaling   = input_scaling
        self.leaking_rate    = leaking_rate
        self.seed_value      = seed_value
        np.random.seed(self.seed_value)
        self.createReservoir()
        
    def resetReservoir(self):
        np.random.seed(self.seed_value)
        self.resetState()
        self.createReservoir()
        
    def resetState(self):
        self.x          = np.zeros((self.d_reservoir, 1))
        self.x_auxilary = np.zeros((self.d_auxilary, 1))
    
    def activationFunction(self, s):
        # take care of numerical instabilities
        s = np.clip(s,1e-3,1.0 -1e-3)
#         a = 1.0/(1.0+np.exp(-s)) # sigmoid activation function
        a = np.tanh(s)
        return a
    
    def createReservoir(self):
        self.Win  = np.random.rand(self.d_reservoir, self.d_input)-0.5
        self.Win  = self.Win * self.input_scaling
        self.W    = np.array(sparse.rand(self.d_reservoir, self.d_reservoir, density=self.sparsity, random_state=self.seed_value).todense())
        self.W[np.where(self.W  > 0)] -=  0.5
        self.W    = self.W * (self.spectral_radius / np.max(np.abs(np.linalg.eig(self.W)[0])))
#         self.Wout  = np.random.rand(d_output, self.d_reservoir+self.d_reservoir)
#         self.Wout    =  self.Wout * (1.0/ np.max(np.abs(self.Wout)))
        
    def updateState(self, u):
        a      = self.activationFunction(np.dot(self.Win, u) + np.dot(self.W, self.x))
        self.x = (1-self.leaking_rate) * self.x + self.leaking_rate * a
    
    def feedImage(self, train_images): 
        N      = train_images.shape[0]
        self.X = np.zeros(((self.d_auxilary + self.d_reservoir), N))
        for i, image in enumerate(train_images):
            self.resetState()
            for row in image:
                self.updateState(row[:,np.newaxis])
            self.x_auxilary = self.x.copy()
            for row in image.T:
                self.updateState(row[:,np.newaxis])
            self.X[:,i] = np.hstack((self.x_auxilary[:,0], self.x[:,0]))
            
    def feedImage_test(self, I): # this function should be customised for image inputs and should be optimized as it consumes the most of computation time
        self.resetState()
        for row in I:
            self.updateState(row[:,np.newaxis])
        self.x_auxilary = self.x.copy()
        for row in I.T:
            self.updateState(row[:,np.newaxis])
    
    def out_init(self, train_images, train_labels): # this function learns Wout using pseudo-inverse  
        transient = min(int(train_images.shape[1] / 10), 100)
        Ytarget = train_labels.T
        self.feedImage(train_images)
        self.beta = 1e-6
        self.Wout = np.dot(Ytarget[transient: ], 
                           np.dot(self.X[:, transient: ].T, np.linalg.inv(np.dot(self.X[:, transient: ], self.X[:, transient: ].T) 
                           + self.beta * np.eye(self.d_auxilary + self.d_reservoir))))
        
    def train(self, train_images, train_labels, learning_rate, class_weights): # learns the ouput weights using batch gradient descent
        N                = train_labels.shape[0]
        Ytarget = train_labels.T
        self.feedImage(train_images)
        predicted_labels = self.predictFromState(self.X)
        E = (predicted_labels - Ytarget)
        D = predicted_labels * (1 - predicted_labels)*E
        loss             = self.weighted_binary_cross_entropy(predicted_labels, Ytarget, class_weights)
        self.Wout        = self.Wout - learning_rate * np.dot(loss, self.X.T)* (1.0/N)
        return (predicted_labels, loss)
        
    def predictFromState(self, x):
        self.y = np.dot(self.Wout, x)
        return 1.0/(1.0+np.exp(-self.y))
        
    def predict(self, image):
        self.feedImage(image)
        self.y = np.dot(self.Wout, self.X)
        return 1.0/(1.0+np.exp(-self.y))
    
    def weighted_binary_cross_entropy(self, output, target, class_weights=None):
        epsilon=1e-3
        output   = np.clip(output, epsilon, 1. - epsilon)
        if class_weights is not None:
            assert len(class_weights) > 1
            log_1 = np.log(output)
            log_0 = np.log(1 - output)
            loss = class_weights[1] * (target * log_1) + class_weights[0] * ((1 - target) * log_0)
        else:
            log_1 = np.log(output)
            log_0 = np.log(1 - output)
            loss = target * log_1 + (1 - target) * log_0
        return -1* np.mean(loss, axis=0) # average class contribution
    
    def echo_state_property(self, x):
        eps =  np.mean(x)
        return eps