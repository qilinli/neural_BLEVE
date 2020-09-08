import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import math
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
    return torch.mean(torch.abs(y_true - y_pred) / y_true) * 100


def train(model, dataset, val_X, val_y, batch_size=128, epochs=1000, epoch_show=10):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        loss_epoch = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            out = model(x)
            loss_iter = loss_fn(out.squeeze(), y)
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()
            loss_epoch += loss_iter / batch_size
        if epoch % epoch_show == 0 or epoch == epochs - 1:
            with torch.no_grad():
                model.eval()
                pred = model(val_X)
                mape = mean_absolute_percentage_error(val_y, pred.squeeze())
                print('Epoch {:03d}: loss={:.6f}, val_mape={:.4f}'.format(epoch, loss_epoch, mape))


def load_data(file, device):
    data = np.load(file)
    train_X = data['train_X']
    train_y = data['train_y']
    val_X = data['val_X']
    val_y = data['val_y']

    train_X = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_X = torch.tensor(val_X, dtype=torch.float32, device=device)
    val_y = torch.tensor(val_y, dtype=torch.float32, device=device)
    dataset = BLEVEDataset(train_X, train_y)

    return dataset, val_X, val_y


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, val_X, val_y = load_data(file='BLEVE_simulated_open.npz', device=device)
    model = MLPNet()
    model.to(device)
    train(model, dataset, val_X, val_y)
