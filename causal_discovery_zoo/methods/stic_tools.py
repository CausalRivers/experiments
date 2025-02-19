import torch
import torch.nn as nn

class DataTrans1(nn.Module):
    def __init__(self, d, lag) -> None:
        super(DataTrans1, self).__init__()
        self.lag = lag
        self.weight_matrix = nn.Parameter(torch.randn(d, lag))
        self.bias_matrix = nn.Parameter(torch.randn(d, lag))
        self.activate = nn.Tanh()

    def forward(self, x):
        x_return = list()
        for i in range(x.shape[1]-self.lag+1):
            tmp_x = self.activate(
                x[:, i:i+self.lag] * self.weight_matrix + self.bias_matrix)
            x_return.append(tmp_x.unsqueeze(0))
        x_return = torch.cat(x_return, dim=0)
        return x_return


class DataTrans(nn.Module):
    def __init__(self, d, lag):
        super(DataTrans, self).__init__()
        self.lag = lag
        self.weight_matrix = nn.Parameter(torch.randn(1, d, lag))
        self.bias_matrix = nn.Parameter(torch.randn(1, d, lag))
        self.activate = nn.Tanh()

    def forward(self, x):
        s = x.shape[0]
        wm = torch.cat([self.weight_matrix for _ in range(s)], dim=0)
        bm = torch.cat([self.bias_matrix for _ in range(s)], dim=0)

        x = self.activate(x*wm+bm)
        return x

# Time Invariance Block with transformation (conv1d)
class CNN(nn.Module):
    def __init__(self, data_dim, max_lag, batch_size) -> None:
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=data_dim,
                              out_channels=data_dim//2,
                              kernel_size=(max_lag+1,))
        self.conv1 = nn.Conv1d(in_channels=data_dim//2,
                               out_channels=1,
                               kernel_size=(1,))
        self.conv2 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=batch_size - max_lag,
                      out_features=data_dim*data_dim*(max_lag+1)),
            nn.PReLU())

        self.lag = max_lag+1
        self.data_trans = DataTrans1(data_dim, self.lag)
        self.data_trans2 = DataTrans(data_dim, self.lag)
        self.activate = nn.Tanh()

    def forward(self, x: torch.Tensor, x_lag: torch.Tensor):
        x_cov = self.conv(x)
        x_conv1 = self.conv1(x_cov)
        graph = self.conv2(x_conv1)
        x_trans = self.data_trans(x_lag)
        x_trans = self.data_trans2(x_trans)
        return graph, x_trans
    

class CNN_lin(nn.Module):
    def __init__(self, data_dim, max_lag, batch_size) -> None:
        super(CNN_lin, self).__init__()
        self.conv = nn.Conv1d(in_channels=data_dim,
                              out_channels=1,
                              kernel_size=(max_lag+1,))
        self.conv1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=batch_size - max_lag,
                      out_features=data_dim*data_dim*(max_lag+1)),
            nn.PReLU())

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        graph = self.conv1(x)
        return graph

