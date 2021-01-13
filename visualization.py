import torch
import torch.nn as nn
import glob
from main import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def scatter_plot_real(df, xmin, xmax):
    # Bar to kpa
    df['output_simulated'] = df['output_simulated'] * 100
    df['output_predicted'] = df['output_predicted'] * 100
    df['flacs'] = df['flacs'] * 100

    # get data in range
    df_subset = df.loc[df['output_simulated'] < xmax]
    df_subset = df_subset.loc[df_subset['output_simulated'] > xmin]
    print("R2_subset: {}".format(r2_score(df_subset['output_simulated'], df_subset['output_predicted'])))
    print("MAPE_subset: {}".format(df_subset['relative_error'].mean(axis=0)))

    sns.scatterplot(data=df_subset,
                    x="output_simulated",
                    y="output_predicted",
                    hue='Status',
                    style='Fluid',
                    palette='pastel',
                    s=100)

    # Add error line
    x_min = df_subset['output_simulated'].min()
    x_max = df_subset['output_simulated'].max()
    xx = np.arange(xmin, xmax, 0.001)
    y_lower = xx * (1 - 0.3)
    y_upper = xx * (1 + 0.3)
    plt.plot(xx, xx, 'r')
    plt.plot(xx, y_lower, 'r', linestyle='--')
    plt.plot(xx, y_upper, 'r', linestyle='--')

    # Add label and axe.limit
    plt.xlabel('Experiment (kPa)')
    plt.ylabel('ANN (kPa)')
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([xmin, xmax])
    plt.grid()

    plt.show()


def scatter_plot(df, xmin, xmax):
    # Bar to kpa
    df['output_simulated'] = df['output_simulated'] * 100
    df['output_predicted'] = df['output_predicted'] * 100

    # get data in range
    df_subset = df.loc[df['output_simulated'] < xmax]
    df_subset = df_subset.loc[df_subset['output_simulated'] > xmin]
    print("R2_subset: {}".format(r2_score(df_subset['output_simulated'], df_subset['output_predicted'])))
    print("MAPE_subset: {}".format(df_subset['relative_error'].mean(axis=0)))

    sns.scatterplot(data=df_subset,
                    x="output_simulated",
                    y="output_predicted",
                    hue='Status',
                    palette='pastel',
                    s=100)

    # Add error line
    x_min = df_subset['output_simulated'].min()
    x_max = df_subset['output_simulated'].max()
    xx = np.arange(xmin, xmax, 0.001)
    y_lower = xx * (1 - 0.3)
    y_upper = xx * (1 + 0.3)
    plt.plot(xx, xx, 'r')
    plt.plot(xx, y_lower, 'r', linestyle='--')
    plt.plot(xx, y_upper, 'r', linestyle='--')

    # Add label and axe.limit
    plt.xlabel('Experiment (kPa)')
    plt.ylabel('ANN (kPa)')
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([xmin, xmax])
    plt.grid()

    plt.show()


def error_analysis(df, feature, target, error):
    sumn = 0
    for i in range(len(target) - 1):
        df_subset = df.loc[df[feature] > target[i]]
        df_subset = df_subset.loc[df_subset[feature] <= target[i + 1]]
        n, _ = df_subset.shape
        sumn += n
        print("==={} < target <= {}, data={}===".format(target[i], target[i + 1], n))
        sump = 0
        for j in range(len(error) - 1):
            df_subset1 = df_subset.loc[df_subset['relative_error'] > error[j]]
            df_subset1 = df_subset1.loc[df_subset1['relative_error'] <= error[j + 1]]
            n_1, _ = df_subset1.shape
            percentage = n_1 / n
            sump += percentage
            print("{} < error <= {},  percentage = {}".format(error[j], error[j + 1], percentage))
        print(" error <= {},  percentage = {}".format(error[-1], sump))
    print("Total data points: {}. Analyzed: {}.".format(df.shape[0], sumn))


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
models_name = glob.glob('models/final_model.pt')
models_name.sort()
model.load_state_dict(torch.load(models_name[-1]), strict=False)
model.eval()


# On training set
train_X = torch.tensor((data['train_X'] - mean) / std, dtype=torch.float32)
train_y = torch.tensor(data['train_y'], dtype=torch.float32)
pred_train = model(train_X)
pred_train = pred_train.squeeze()
print("MAPE_train: {}".format(mean_absolute_percentage_error(train_y, pred_train)))
print("R2_train: {}".format(r2_score(data['train_y'], pred_train.detach().numpy())))

df_train = pd.DataFrame(data['train_X'], columns=columns)
df_train = df_train.assign(output_simulated=data['train_y'])
df_train = df_train.assign(output_predicted=pred_train.detach().numpy())
df_train = df_train.assign(relative_error=np.abs(pred_train.detach().numpy()
                                               - data['train_y'])/data['train_y'] * 100)

# On val set
val_X = torch.tensor((data['val_X'] - mean) / std, dtype=torch.float32)
val_y = torch.tensor(data['val_y'], dtype=torch.float32)
pred_val = model(val_X)
pred_val = pred_val.squeeze()
print("MAPE_val: {}".format(mean_absolute_percentage_error(val_y, pred_val)))
print("R2_val: {}".format(r2_score(data['val_y'], pred_val.detach().numpy())))

df_val = pd.DataFrame(data['val_X'], columns=columns)
df_val = df_val.assign(output_simulated=data['val_y'])
df_val = df_val.assign(output_predicted=pred_val.detach().numpy())
df_val = df_val.assign(relative_error=np.abs(pred_val.detach().numpy()
                                               - data['val_y'])/data['val_y'] * 100)

# On test set
test_X = torch.tensor((data['test_X'] - mean) / std, dtype=torch.float32)
test_y = torch.tensor(data['test_y'], dtype=torch.float32)
pred_test = model(test_X)
pred_test = pred_test.squeeze()
print("MAPE_test: {}".format(mean_absolute_percentage_error(test_y, pred_test)))
print("R2_test: {}".format(r2_score(data['test_y'], pred_test.detach().numpy())))

df_test = pd.DataFrame(data['test_X'], columns=columns)
df_test = df_test.assign(output_simulated=data['test_y'])
df_test = df_test.assign(output_predicted=pred_test.detach().numpy())
df_test = df_test.assign(relative_error=np.abs(pred_test.detach().numpy()
                                               - data['test_y'])/data['test_y'] * 100)
# df_test.to_excel("output.xlsx", sheet_name='simulated_data')

# scatter_plot(df_test, 100, 500)

# Target pressure error analysis
# target = [0, 0.05, 0.1, 0.2, 0.5, 5]
# error = [0, 5, 10, 20, 30]
# feature = 'output_simulated'
#
# # BLEVE distance error analysis
# target = [0, 10, 20, 30, 40, 50]
# error = [0, 5, 10, 20, 30]
# feature = 'Distance to BLEVE'
#
# # Liquid ratio error analysis
# target = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
# error = [0, 5, 10, 20, 30]
# feature = 'Liquid ratio'
# #
# #  BLEVE distance error analysis
# target = [0, 0.3, 0.4, 0.6, 1.0, 2.0]
# error = [0, 5, 10, 20, 30]
# feature = 'Gas height  (m)'
#
# error_analysis(df_test, feature, target, error)

# LOAD real data
real_test_X = data['real_test_X']
real_test_y = data['real_test_y']

# Pre-processing for prediction
real_test_X = torch.tensor((real_test_X - mean) / std, dtype=torch.float32)
real_test_y = torch.tensor(real_test_y, dtype=torch.float32)
real_pred_test = model(real_test_X)
real_pred_test = real_pred_test.squeeze()
print("MAPE_real_test: {}".format(mean_absolute_percentage_error(real_test_y, real_pred_test)))
print("R2_real_test: {}".format(r2_score(data['real_test_y'], real_pred_test.detach().numpy())))


df_test_real = pd.DataFrame(data['real_test_X'], columns=columns)
df_test_real = df_test_real.assign(output_simulated=data['real_test_y'])
real_pred_test = real_pred_test.detach().numpy()
df_test_real = df_test_real.assign(output_predicted=real_pred_test)
df_test_real = df_test_real.assign(relative_error=np.abs(real_pred_test -
                                                         data['real_test_y'])/data['real_test_y'] * 100)
df = pd.read_excel('data_simulated_real_Butane_Propane_T4.xlsx', 'real_noVal')
df_id_fluid = df.iloc[:, 0:2]
df_test_real.loc[df_test_real['Status'] == 0, 'Status'] = 'Saturated'
df_test_real.loc[df_test_real['Status'] == 1, 'Status'] = 'Superheated'
df_test_real = pd.concat([df_test_real, df_id_fluid], axis=1)

df = pd.read_excel('data_simulated_real_Butane_Propane_T4.xlsx', 'Target_FLACS_ANN')
print('FLACS vs ANN: ', np.mean(np.abs(df['flacs'] - df['ann'])/df['flacs'] * 100))
print('FLACS vs Exp: ', np.mean(np.abs(df['flacs'] - df['target'])/df['target'] * 100))
print('ANN vs Exp: ', np.mean(np.abs(df['ann'] - df['target'])/df['target'] * 100))
df_test_real = pd.concat([df_test_real, df['flacs']], axis=1)
# df_test_real.to_excel("output_real_noVal.xlsx", sheet_name='real_no_Val')

scatter_plot_real(df_test_real, 0, 12)

# BLEVE distance error analysis
# target = [9, 20, 40, 100]
# error = [0, 10, 20, 30, 50]
# feature = 'Distance to BLEVE'

# error_analysis(df_test_real, feature, target, error)
# print(df_test_real['relative_error'])


