import pandas as pd
import numpy as np
from random import shuffle
from sklearn.preprocessing import StandardScaler


def load_data(file):
    # Load the excel and extract one of the sheets
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, 'simulated')

    # Shuffle the dataset
    df = df.sample(frac=1)

    # Converte a str column to int
    df.loc[df['Status'] == 'Subcooled', 'Status'] = 0
    df.loc[df['Status'] == 'Superheated', 'Status'] = 1

    # Col 0 is ID, Col 1-10 are features
    X = df.iloc[:, 1:11]
    X['Status'] = X['Status'].astype('float32')

    # Cols 11-56 are 46 sensors with varying distance to BLEVE
    Y = df.iloc[:, 11:]
    XY = []
    for i in range(Y.shape[0]):
        cols = [x for x in list(Y.columns.values)]

        for j in range(len(cols)):
            x = X.iloc[i, :].tolist()
            x.append(int(cols[j]))  # add label of col as feature "Distance from BLEVE"
            x.append(Y.iloc[i, cols[j] - 5])  # add target values
            XY.append(x)

    columns = list(X.columns.values)
    columns.append('Distance from BLEVE')
    columns.append('target')

    data = pd.DataFrame(XY, columns=columns)
    missing_values = data.isnull().values.any()
    print(data.columns[data.isnull().any()])
    if missing_values:
        print("===There is Missing value===")

    target = data["target"]
    data.drop("target", axis=1, inplace=True)

    # dataset split, 70% training 15 validation 15 testing
    n_train = int(data.shape[0] * 0.7)
    n_val = int(data.shape[0] * 0.85)

    train_X = data[:n_train].to_numpy()
    train_y = target[:n_train].to_numpy()
    val_X = data[n_train:n_val].to_numpy()
    val_y = target[n_train:n_val].to_numpy()
    test_X = data[n_val:].to_numpy()
    test_y = target[n_val:].to_numpy()

    train_X = train_X.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_X = val_X.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_X = test_X.astype(np.float32)
    test_y = test_y.astype(np.float32)

    # real data
    df = pd.read_excel(xls, 'real_noVal')
    df = df.iloc[:, 2:]     # The first col is ID, second is fluid
    real_data = df.to_numpy()
    real_test_X = real_data[:, :-1]
    real_test_y = real_data[:, -1]

    # Data preprocessing
    scaler = StandardScaler().fit(train_X)
    mean_X = scaler.mean_
    std_X = scaler.scale_

    # Normalization moved to main.py
    # train_X = scaler.transform(train_X)
    # val_X = scaler.transform(val_X)
    # test_X = scaler.transform(test_X)
    # real_test_X = scaler.transform(real_test_X)

    print(train_X.shape, val_X.shape, test_X.shape, real_test_X.shape)
    return train_X, train_y, val_X, val_y, test_X, test_y, real_test_X, real_test_y, mean_X, std_X


if __name__ == '__main__':
    train_X, train_y, val_X, val_y, test_X, test_y, real_test_X, real_test_y, mean, std = load_data(
        'data_simulated_real_Butane_Propane_T4.xlsx')
    np.savez('BLEVE_Butane_Propane', train_X=train_X, train_y=train_y,
             val_X=val_X, val_y=val_y, test_X=test_X, test_y=test_y,
             real_test_X=real_test_X, real_test_y=real_test_y, mean=mean, std=std)