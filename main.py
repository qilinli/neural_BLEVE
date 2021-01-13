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
import glob

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
        self.net.add_module('out1', nn.Softplus(beta=5))

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
    return torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true)) * 100


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, dataset, val_X, val_y, batch_size=512, epochs=500, epoch_show=1, weight_decay=1e-5, momentum=0.9):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=weight_decay)
    min_lr = 1e-4
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.1,
                                                     patience=50,
                                                     threshold=0,
                                                     min_lr=min_lr,
                                                     verbose=True)
    loss_fn = nn.MSELoss(reduction='mean')

    writer = SummaryWriter('runs/batchSize/linear_hidden={}_neurons={}_{}_batch={:04d}_bn={}_p={:.1f}_mom={}_l2={}'.format(
        len(model.features)-1, model.features[-1], model.activation_fn_name, batch_size, model.bn, model.p, momentum,
        weight_decay
    ))

    best_val_mape = 100
    for epoch in range(epochs):
        # if get_lr(optimizer) < 2 * min_lr:
        #     break
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
                      'best_val_mape={:.4f}'.format(epoch, loss_epoch/(i+1), loss_val, mape, best_val_mape), end='  ')
                if mape < best_val_mape:
                    model_name = 'running_best_model.pt'
                    print('Val_mape improved from {:.4f} to {:.4f}, saving model to {}'.format(
                        best_val_mape, mape, model_name), end=' ')
                    best_val_mape = mape
                    torch.save(model.state_dict(), 'models/' + model_name)
        writer.add_scalar("loss/train", loss_epoch/(i+1), epoch)
        writer.add_scalar("loss/val", loss_val, epoch)
        writer.add_scalar("mape/val", mape, epoch)
    torch.save(model.state_dict(), 'models/final_model.pt')
    writer.add_scalar("mape/best_val", best_val_mape)
    return writer


def test(model, models_dir, test_X, test_y, real_test_X, real_test_y):
    model.eval()
    models_name = glob.glob(models_dir + 'running_best_model.pt')
    models_name.sort()
    model.load_state_dict(torch.load(models_name[-1]), strict=False)
    pred = model(test_X)
    loss = nn.MSELoss()(pred.squeeze(), test_y)
    mape = mean_absolute_percentage_error(test_y, pred.squeeze())
    print('\nloss_test: {:.6f}, mape_test:{:.4f}'.format(loss, mape))
    # writer.add_scalar('mape/test', mape)

    pred = model(real_test_X)
    print(pred.squeeze())
    print(real_test_y)
    print(torch.abs(real_test_y - pred.squeeze()) / real_test_y * 100)
    loss = nn.MSELoss()(pred.squeeze(), real_test_y)
    mape = mean_absolute_percentage_error(real_test_y, pred.squeeze())
    print('\nloss_real_test: {:.6f}, mape_real_test:{:.4f}'.format(loss, mape))

    import json
    output = torch.abs(real_test_y - pred.squeeze()) / real_test_y * 100
    output = output.detach().tolist()
    output = [float('%.1f' % (x)) for x in output]
    with open("prediction_on_real_data.txt", "a") as outfile:
        json.dump(output, outfile)
        outfile.write('  ' + str(np.mean(output)))
        outfile.write('\n')


def load_data(file, device):
    data = np.load(file)
    train_X = data['train_X']
    train_y = data['train_y']
    val_X = data['val_X']
    val_y = data['val_y']
    test_X = data['test_X']
    test_y = data['test_y']
    real_test_X = data['real_test_X']
    real_test_y = data['real_test_y']
    mean = data['mean']
    std = data['std']

    # from sklearn.preprocessing import PowerTransformer
    # pt = PowerTransformer(method='box-cox', standardize=True)
    # train_y = pt.fit_transform(train_y.reshape(-1,1)).squeeze()
    # val_y = pt.transform(val_y.reshape(-1,1)).squeeze()
    # test_y = pt.transform(test_y.reshape(-1,1)).squeeze()
    # real_test_y = pt.transform(real_test_y.reshape(-1,1)).squeeze()

    train_X, val_X, test_X, real_test_X = map(lambda x: (x - mean) / std,
                                              [train_X, val_X, test_X, real_test_X])

    train_X, val_X, test_X, real_test_X = map(lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                                              [train_X, val_X, test_X, real_test_X])
    train_y, val_y, test_y, real_test_y = map(lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                                              [train_y, val_y, test_y, real_test_y])

    dataset = BLEVEDataset(train_X, train_y)

    return dataset, val_X, val_y, test_X, test_y, real_test_X, real_test_y


if __name__ == '__main__':
    for i in range(1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset, val_X, val_y, test_X, test_y, real_test_X, real_test_y = load_data(
            file='BLEVE_Butane_Propane.npz', device=device)
        print(len(val_X))

        activation_list = ['mish']
        bn_list = [0]
        p_list = [0]
        batchSize_list = [512]
        # feature_list = [[val_X.shape[1], 64], [val_X.shape[1], 128], [val_X.shape[1], 256], [val_X.shape[1], 512],
        #                 [val_X.shape[1], 64, 64], [val_X.shape[1], 128, 128], [val_X.shape[1], 256, 256], [val_X.shape[1], 512, 512]
        #                 [val_X.shape[1], 64, 64, 64], [val_X.shape[1], 128, 128, 128], [val_X.shape[1], 256, 256, 256], [val_X.shape[1], 512, 512, 512],
        #                 [val_X.shape[1], 64, 64, 64, 64], [val_X.shape[1], 128, 128, 128, 128], [val_X.shape[1], 256, 256, 256, 256], [val_X.shape[1], 512, 512, 512, 512]]
        feature_list =[[val_X.shape[1], 256, 256, 256]]
        momentum_list = [0.9]
        weight_decay_list = [1e-6]

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
                                    test(model, 'models/', test_X, test_y, real_test_X, real_test_y)
