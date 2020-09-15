import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import math
from torch import sigmoid, tanh, relu
from torch.utils.tensorboard import SummaryWriter
from mish import Mish
from ranger import Ranger


class BLEVEDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]


class MLPNet(nn.Module):
    def __init__(self, features=[10, 256, 256], bn=False, activation_fn='mish', p=0):
        super().__init__()
        self.net = nn.Sequential()
        if activation_fn == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation_fn == 'mish':
            self.activation_fn = Mish()
        else:
            self.activation_fn = nn.ReLU(inplace=True)

        self.features = features
        self.bn = bn
        self.activation_fn_name = activation_fn
        self.p = p

        for layer in range(1, len(features)):
            self.net.add_module('fc%d' % layer, nn.Linear(features[layer - 1], features[layer]))
            if bn:
                self.net.add_module('bn%d' % layer, nn.BatchNorm1d(features[layer]))
            self.net.add_module('sig%d' % layer, self.activation_fn)
            if p > 0:
                self.net.add_module('dp%d' % layer, nn.Dropout(p))
        self.net.add_module('out', nn.Linear(features[-1], 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_normal_(m.weight.data)
                m.bias.data = torch.nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data = nn.init.normal_(m.weight.data, mean=1, std=0.02)
                m.bias.data = torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.net(x)
        return x


def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred) / y_true) * 100


def train(model, dataset, val_X, val_y, batch_size=512, epochs=500, epoch_show=1, weight_decay=1e-5, momentum=0.99):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # optimizer = Ranger(model.parameters(), weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.1,
                                                     patience=50,
                                                     threshold=0,
                                                     verbose=True)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter('runs/Mish/hidden={}_neurons={}_{}_batch={:04d}_bn={}_p={:.1f}_mom={}_l2={}'.format(
        len(model.features)-1, model.features[-1], model.activation_fn_name, batch_size, model.bn, model.p, momentum,
        weight_decay
    ))

    best_val_mape = 100
    for epoch in range(epochs):
        model.train()
        loss_epoch = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            out = model(x)
            loss_iter = loss_fn(y, out.squeeze())
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()
            loss_epoch += loss_iter
        if epoch % epoch_show == 0 or epoch == epochs - 1:
            with torch.no_grad():
                model.eval()
                pred = model(val_X)
                loss_val = loss_fn(val_y, pred.squeeze())
                mape = mean_absolute_percentage_error(val_y, pred.squeeze())
                scheduler.step(best_val_mape)
                print('\nEpoch {:03d}: loss_train={:.6f}, loss_val={:.6f}, val_mape={:.4f}, '
                      'best_val_mape={:.4f}'.format(epoch, loss_epoch, loss_val, mape, best_val_mape), end='  ')
                if mape < best_val_mape:
                    model_name = 'best_model.pt'
                    print('Val_mape improved from {:.4f} to {:.4f}, saving model to {}'.format(
                        best_val_mape, mape, model_name), end=' ')
                    best_val_mape = mape
                    torch.save(model.state_dict(), 'models/' + model_name)
        writer.add_scalar("loss/train", loss_epoch, epoch)
        writer.add_scalar("loss/val", loss_val, epoch)
        writer.add_scalar("mape/val", mape, epoch)
    writer.add_scalar("mape/best_val", best_val_mape)
    return writer


def test(writer, model, models_dir, test_X, test_y):
    model.eval()
    import glob
    models_name = glob.glob(models_dir + '*.pt')
    models_name.sort()
    model.load_state_dict(torch.load(models_name[-1]), strict=False)
    pred = model(test_X)
    loss = nn.MSELoss()(pred.squeeze(), test_y)
    mape = mean_absolute_percentage_error(test_y, pred.squeeze())
    print('\nloss_test: {:.6f}, mape_test:{:.4f}'.format(loss, mape))
    writer.add_scalar('mape/test', mape)


def load_data(file, device):
    data = np.load(file)
    train_X = data['train_X']
    train_y = data['train_y']
    val_X = data['val_X']
    val_y = data['val_y']
    test_X = data['test_X']
    test_y = data['test_y']

    train_X = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_X = torch.tensor(val_X, dtype=torch.float32, device=device)
    val_y = torch.tensor(val_y, dtype=torch.float32, device=device)
    test_X = torch.tensor(test_X, dtype=torch.float32, device=device)
    test_y = torch.tensor(test_y, dtype=torch.float32, device=device)

    dataset = BLEVEDataset(train_X, train_y)

    return dataset, val_X, val_y, test_X, test_y


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, val_X, val_y, test_X, test_y = load_data(file='BLEVE_simulated_open.npz', device=device)

    activation_list = ['mish']
    bn_list = [0]
    p_list = [0, 0.1]
    batchSize_list = [256, 512, 1024]
    feature_list = [[10, 256, 256], [10, 256, 256, 256]]
    momentum_list = [0.99]
    weight_decay_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

    for activation_fn in activation_list:
        for bn in bn_list:
            for p in p_list:
                for batch_size in batchSize_list:
                    for features in feature_list:
                        for momentum in momentum_list:
                            for weight_decay in weight_decay_list:
                                print("{}_bn={}_p={}_batchSize={}_feature={}_mom={}_l2={}".format(
                                    activation_fn, bn, p, batch_size, features, momentum, weight_decay
                                ))
                                model = MLPNet(features=features, activation_fn=activation_fn, bn=bn, p=p)
                                model.to(device)
                                writer = train(model, dataset, val_X, val_y, batch_size=batch_size,
                                               momentum=momentum, weight_decay=weight_decay)
                                test(writer, model, 'models/', test_X, test_y)
