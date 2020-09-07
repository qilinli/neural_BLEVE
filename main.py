import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import math
from load_data_from_excel import load_data
from torch import sigmoid, tanh, relu


class BLEVEDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]


class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = sigmoid(self.fc1(x))
        x = sigmoid(self.fc2(x))
        x = self.out(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m)
    if isinstance(m, nn.BatchNorm):
        nn.init.xavier_normal_(m)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / y_true) * 100


def train(model):
    train_X, val_X, train_y, val_y = load_data('uniform_synthetic_data_Butane_N=5000_D=12 - T2.xlsx')
    dataset = BLEVEDataset(train_X, train_y)
    train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    for epoch in range(1000):
        model.train()
        loss_epoch = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            # x.to(device)
            # y.to(device)
            out = model(x)
            loss_iter = loss_fn(out.squeeze(), y)
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()
            loss_epoch += loss_iter / 128
        if epoch % 1 == 0 or epoch == 999:
            with torch.no_grad():
                model.eval()
                pred = model(val_X)
                mape = loss_fn(val_y, pred.squeeze())
                print('Epoch {:03d}: loss={:.6f}, val_mape={:.4f}'.format(epoch, loss_epoch, mape))


if __name__ == '__main__':
    model = MLPNet()
    train(model)
