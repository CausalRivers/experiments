import random
import numpy as np
import heapq
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


"""Adapted from https://github.com/M-Nauta/TCDF"""


class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions, 
    so it is implemented (with some inefficiency) by simply using a 
    standard convolution with zero padding on both sides, and chopping
    off the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FirstBlock(nn.Module):
    def __init__(self, target, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()
        
        self.target = target
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)      
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        
    def forward(self, x):
        out = self.net(x)
        return self.relu(out)    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
       
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        

    def forward(self, x):
        out = self.net(x)
        return self.relu(out+x) #residual connection

class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01) 
        
    def forward(self, x):
        out = self.net(x)
        return self.linear(out.transpose(1,2)+x.transpose(1,2)).transpose(1,2) #residual connection

class DepthwiseNet(nn.Module):
    def __init__(self, target, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l==0:
                layers += [FirstBlock(target, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            elif l==num_levels-1:
                layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ADDSTCN(nn.Module):
    def __init__(self, target, input_size, num_levels, kernel_size, device, dilation_c):
        super(ADDSTCN, self).__init__()

        self.target=target
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)

        self._attention = torch.ones(input_size,1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = torch.nn.Parameter(self._attention.data)
        
        self.dwn = self.dwn.to(device)
        self.pointwise = self.pointwise.to(device)
        self._attention = self._attention.to(device)
                  
    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x):
        y1=self.dwn(x*F.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1) 
        return y1.transpose(1,2)

def train(epoch, traindata, traintarget, modelname, optimizer,log_interval,epochs):
    """Trains model by performing one epoch and returns attention scores and loss."""

    modelname.train()
    x, y = traindata[0:1], traintarget[0:1]
        
    optimizer.zero_grad()
    epochpercentage = (epoch/float(epochs))*100
    output = modelname(x)

    attentionscores = modelname.fs_attention
    
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()

    if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
        print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

    return attentionscores.data, loss


# def preparedata(file, target):
#     """Reads data from csv file and transforms it to two PyTorch tensors: dataset x and target time series y that has to be predicted."""
#     df_data = pd.read_csv(file)
#     df_y = df_data.copy(deep=True)[[target]]
#     df_x = df_data.copy(deep=True)
#     df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
#     df_yshift[target]=df_yshift[target].fillna(0.)
#     df_x[target] = df_yshift
#     data_x = df_x.values.astype('float32').transpose()    
#     data_y = df_y.values.astype('float32').transpose()
#     data_x = torch.from_numpy(data_x)
#     data_y = torch.from_numpy(data_y)

#     x, y = Variable(data_x), Variable(data_y)
#    return x, y

def prep(X, idx):
    y = X[:,idx].reshape(-1,1)
    X_ = X.copy()

    X_idx = X_[:-1,idx].copy()
    X_[:,idx] = 0
    X_[1:,idx] = X_idx

    data_x = X_.astype('float32').transpose()    
    data_y = y.astype('float32').transpose()
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    x, y = Variable(data_x), Variable(data_y)
    return x, y

def findcauses(data, target, device, epochs, kernel_size, layers, 
               log_interval, lr, optimizername, seed, dilation_c, significance):
    """Discovers potential causes of one target time series, validates these potential causes with PIVM and discovers the corresponding time delays"""

    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed)
    
    X_train, Y_train = prep(data, target)
    X_train = X_train.unsqueeze(0).contiguous()
    Y_train = Y_train.unsqueeze(2).contiguous()

    input_channels = X_train.size()[1]
          
    model = ADDSTCN(target, input_channels, layers, kernel_size=kernel_size, device=device, dilation_c=dilation_c)
    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    scores, firstloss = train(1, X_train, Y_train, model, optimizer,log_interval,epochs)
    firstloss = firstloss.cpu().data.item()
    for ep in range(2, epochs+1):
        scores, realloss = train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
    realloss = realloss.cpu().data.item()
    
    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
    
    #attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i]>1.:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i]<1.: #tau should be greater or equal to 1, so only consider scores >= 1
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)
        
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index<((len(s)-1)/2): #gap should be in first half
                if index>0:
                    ind=index #gap should have index > 0, except if second score <1
                    break
        if ind<0:
            ind = 0
                
        potentials = indices[:ind+1].tolist()
    print("Potential causes: ", potentials)
    validated = copy.deepcopy(potentials)
    
    #Apply PIVM (permutes the values) to check if potential cause is true cause
    for idx in potentials:
        random.seed(seed)
        X_test2 = X_train.clone().cpu().numpy()
        random.shuffle(X_test2[:,idx,:][0])
        shuffled = torch.from_numpy(X_test2)
        shuffled=shuffled.to(device)
        model.eval()
        output = model(shuffled)
        testloss = F.mse_loss(output, Y_train)
        testloss = testloss.cpu().data.item()
        
        diff = firstloss-realloss
        testdiff = firstloss-testloss

        if testdiff>(diff*significance): 
            validated.remove(idx) 
    
    weights = []
    
    #Discover time delay between cause and effect by interpreting kernel weights
    for layer in range(layers):
        weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], model.dwn.network[layer].net[0].weight.size()[2])
        weights.append(weight)

    causeswithdelay = dict()    
    for v in validated: 
        totaldelay=0 
        for k in range(len(weights)):
            w=weights[k]
            row = w[v]
            twolargest = heapq.nlargest(2, row)
            m = twolargest[0]
            m2 = twolargest[1]
            if m > m2:
                index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
            else:
                #take first filter
                index_max=0
            delay = index_max *(dilation_c**k)
            totaldelay+=delay
        if target != v:
            causeswithdelay[(target, v)]=totaldelay
        else:
            causeswithdelay[(target, v)]=totaldelay+1
    print("Validated causes: ", validated)

    return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist(), weights[0].cpu().detach().numpy()



# direct weight usage
def patch_weights(all_weights):
    # we in the beginning deleted the current timestep of the target variable
    # here we do this reversely. This also ensures that we do not discover instantatnious loops
    all_weights_ = {}
    for v in all_weights:
        all_weights_[v] = all_weights[v].copy()
        all_weights_[v][v, :-1] = all_weights[v][v,1:]
        all_weights_[v][v,-1] = 0

        # for each target, we have to flip it around
        # here lag -1 is lag zero
        #TODO normalize vektorwise 
        all_weights_[v] = np.flip(all_weights_[v], axis=1)

    return all_weights_

def convert_weights_to_window_dag(all_weights):
    all_weights = patch_weights(all_weights)
    num_vars = len(all_weights)
    
    # effect x cause x lags
    weights = np.stack([all_weights[i] for i in range(num_vars)], axis=0)
    return weights

