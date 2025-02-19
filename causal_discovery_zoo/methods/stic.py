
import numpy as np
import torch
from torch import nn
from methods.stic_tools import CNN
from tqdm import tqdm


def stic(data, cfg, device):
    data_dim = data.shape[1]
    T = data.shape[0]
    max_lag = cfg.max_lag
    batch_size = cfg.batch_size
    lr = cfg.lr
    lr_decay_stepsize = cfg.lr_decay_stepsize
    epochs = cfg.epochs
    tqdm_bar = cfg.tqdm_bar

    data_torch = torch.from_numpy(data).to(device).float()
    cnn = CNN(data_dim, max_lag, batch_size).to(device)

    optim = torch.optim.SGD(cnn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if tqdm_bar:
        iterator = tqdm(range(epochs), total=epochs)
    else:
        iterator = range(epochs)

    for epoch in iterator:
        loss_sum = 0

        if epoch % lr_decay_stepsize == 0 and epoch != 0:
            for g in optim.param_groups:
                g['lr'] = g["lr"]/10

        for i in range(max_lag, T-batch_size, max_lag):
            x = data_torch[i:i+batch_size, :]
            x_T = x.T.unsqueeze(0)

            x_lag = data_torch[i-max_lag:i+batch_size, :].T
            conv_x, x_trans = cnn(x_T, x_lag)
            conv_x = conv_x.reshape(
                (data_dim, data_dim, (max_lag+1)))

            # Avoid self cycle
            for k in range(data_dim):
                conv_x[k, k, 0] = 0

            # Hadamard product
            y_hat = torch.zeros_like(x)
            for j in range(data_dim):
                for lag in range(max_lag+1):
                    conv_x_j_lag = conv_x[:, j:j+1, lag]
                    y_hat[:, j] += torch.mm(x_trans[:, :, max_lag-lag],
                                        conv_x_j_lag).squeeze(1)

            loss = loss_fn(y_hat, x)
            loss_sum += loss
            optim.zero_grad()
            loss.backward()
            optim.step()

    res = np.abs(conv_x.detach().cpu().numpy())
    return res


# Adapted from https://github.com/HITshenrj/STIC
def stic_baseline(data_sample, cfg):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    # data preprocessing
    # expect data in dataframe where rows are timesteps
    # and columns are variables
    data = data_sample.to_numpy()

    res = stic(data, cfg, device)

    # res is 3d array with
    # cause x effect x lag

    # we reorder it to Gideons format
    # effect, cause, lag
    res = np.moveaxis(res, 1, 0)
    return res




