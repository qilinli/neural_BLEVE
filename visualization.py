import torch
import torch.nn as nn
import glob
from main import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Name of features
columns = ['Tank failure Pressure (bar)',
           'Liquid ratio',
           'Tank width (m)',
           'Tank length (m)',
           'Tank height (m)',
           'Height of BLEVE (m)',
           'Vapour temerature (K)',
           'Liquid temerature (K)',
           'Status',
           'Gas height  (m)',
           'Distance to BLEVE']

# Check the performance
data = np.load('BLEVE_Butane_Propane.npz')
mean = data['mean']
std = data['std']

model = MLPNet(features=[mean.shape[0], 256, 256, 256], activation_fn='mish')
models_name = glob.glob('models/running_best_model.pt')
models_name.sort()
model.load_state_dict(torch.load(models_name[-1]), strict=False)
model.eval()

train_X = torch.tensor((data['train_X'] - mean) / std, dtype=torch.float32)
train_y = torch.tensor(data['train_y'], dtype=torch.float32)
pred_train = model(train_X)
pred_train = pred_train.squeeze()
print("MAPE_train: {}".format(mean_absolute_percentage_error(train_y, pred_train)))
print("R2_train: {}".format(r2_score(data['train_y'], pred_train.detach().numpy())))

df_train = pd.DataFrame(data['train_X'], columns=columns)
df_train = df_train.assign(output_simulated=data['train_y'])
df_train = df_train.assign(output_predicted=pred_train.detach().numpy())

val_X = torch.tensor((data['val_X'] - mean) / std, dtype=torch.float32)
val_y = torch.tensor(data['val_y'], dtype=torch.float32)
pred_val = model(val_X)
pred_val = pred_val.squeeze()
print("MAPE_val: {}".format(mean_absolute_percentage_error(val_y, pred_val)))
print("R2_val: {}".format(r2_score(data['val_y'], pred_val.detach().numpy())))

df_val = pd.DataFrame(data['val_X'], columns=columns)
df_val = df_val.assign(output_simulated=data['val_y'])
df_val = df_val.assign(output_predicted=pred_val.detach().numpy())

test_X = torch.tensor((data['test_X'] - mean) / std, dtype=torch.float32)
test_y = torch.tensor(data['test_y'], dtype=torch.float32)
pred_test = model(test_X)
pred_test = pred_test.squeeze()
print("MAPE_test: {}".format(mean_absolute_percentage_error(test_y, pred_test)))
print("R2_test: {}".format(r2_score(data['test_y'], pred_test.detach().numpy())))

# LOAD real data
real_test_X = data['real_test_X']
real_test_y = data['real_test_y']

real_test_X = torch.tensor((real_test_X - mean) / std, dtype=torch.float32)
real_test_y = torch.tensor(real_test_y, dtype=torch.float32)
real_pred_test = model(real_test_X)
real_pred_test = real_pred_test.squeeze()
print("MAPE_real_test: {}".format(mean_absolute_percentage_error(real_test_y, real_pred_test)))
print("R2_real_test: {}".format(r2_score(data['real_test_y'], real_pred_test.detach().numpy())))


# df_test = pd.DataFrame(data['test_X'], columns=columns)
# df_test = df_test.assign(output_simulated=data['test_y'])
# df_test = df_test.assign(output_predicted=pred_test.detach().numpy())
# df_test.to_excel("output.xlsx", sheet_name='simulated_data')
#
# df_test_real = pd.DataFrame(real_data[:, :-1], columns=columns)
# df_test_real = df_test_real.assign(output_simulated=real_data[:, -1])
# real_pred_test = real_pred_test.detach().numpy()
# df_test_real = df_test_real.assign(output_predicted=real_pred_test)
# df_test_real = df_test_real.assign(relative_error=np.abs(real_pred_test -
#                                                          data['real_test_y'])/data['real_test_y'] * 100)
# df_test_real.to_excel("output_real_data.xlsx", sheet_name='real_data')
#
# df_test_small = df_train.loc[df_train['output_simulated'] < 0.1]
#df_test_small = df_test_small.loc[df_test_small['output_simulated'] < 1]

# sns.scatterplot(data=df_test_small,
#                 x='output_predicted',
#                 y="output_simulated",
#                 hue='Status',
#                 s=10)
# x_min = df_test_small['output_simulated'].min()
# x_max = df_test_small['output_simulated'].max()
# xx = np.arange(x_min, x_max, 0.001)
#
# sns.lineplot(x=xx, y=xx, color='r')

#sns.lmplot(data=df_test, x="output_predicted", y="output_simulated", hue='Distance to BLEVE')
# plt.show()