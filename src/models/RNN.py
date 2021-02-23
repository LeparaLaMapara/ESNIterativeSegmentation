import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of ESN
"""

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

"""
Implementation of RNN
"""
class LSTM(nn.Module):
    def __init__(self, lstm_input_size=512, lstm_hidden_size=512, lstm_num_layers=3,
                num_classes=100, hidden1=256, drop_p=0.0):
        super(LSTM, self).__init__()
        # network params
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.drop_p = drop_p

        # network architecture
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, x):
        # RNN
        hidden=None
        # use faster code paths
        self.lstm.flatten_parameters()
        # print(x.shape)
        # batch first: (batch, seq, feature)
        out, hidden = self.lstm(x, hidden)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc2(out)

        return out



"""
Implementation of LSTM
"""
class LSTM(nn.Module):
    def __init__(self, lstm_input_size=512, lstm_hidden_size=512, lstm_num_layers=3,
                num_classes=100, hidden1=256, drop_p=0.0):
        super(LSTM, self).__init__()
        # network params
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.drop_p = drop_p

        # network architecture
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, x):
        # LSTM
        hidden = None
        # use faster code paths
        self.lstm.flatten_parameters()
        # print(x.shape)
        # batch first: (batch, seq, feature)
        out, hidden = self.lstm(x, hidden)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc2(out)

        return out


"""
Implementation of GRU
"""
class GRU(nn.Module):
    def __init__(self, gru_input_size=512, gru_hidden_size=512, gru_num_layers=3,
                num_classes=100, hidden1=256, drop_p=0.0):
        super(GRU, self).__init__()
        # network params
        self.gru_input_size = gru_input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.drop_p = drop_p

        # network architecture
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.gru_hidden_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, x):
        # GRU
        hidden = None
        # use faster code paths
        self.gru.flatten_parameters()
        # print(x.shape)
        # batch first: (batch, seq, feature)
        out, hidden = self.gru(x, hidden)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc2(out)

        return out