import pandas as pd
import numpy as np
from random import shuffle
from sklearn.preprocessing import StandardScaler


def load_data(file):
    # Load the excel and extract one of the sheets
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, 'Inputs(clean)')

    # Shuffle the dataset
    df = df.sample(frac=1)

    # Converte a str column to int
    df.loc[df['Status'] == 'Subcooled', 'Status'] = 0
    df.loc[df['Status'] == 'Superheated', 'Status'] = 1

    # Col 0 is ID, Col 1-9 are features
    X = df.iloc[:, 1:10]
    X['Status'] = X['Status'].astype('float32')

    # Col 10 is the minimum distance, Cols 11-56 are 46 sensors with varying distance to BLEVE
    Y = df.iloc[:, 10:]
    XY = []
    for i in range(Y.shape[0]):
        cols = [x for x in list(Y.columns[1:].values)
                if int(x) >= Y['Starting distance (m)'].iloc[i]]    # Extract cols that satisfy "minimal distance"
        shuffle(cols)

        for j in range(len(cols)):
            x = X.iloc[i, :].tolist()
            x.append(int(cols[j]))  # add label of col as feature "Distance from BLEVE"
            x.append(Y.iloc[i, cols[j] - 4])  # add target values
            XY.append(x)

    columns = list(X.columns.values)
    columns.append('Distance from BLEVE')
    columns.append('target')

    data = pd.DataFrame(XY, columns=columns)
    missing_values = data.isnull().values.any()
    if missing_values:
        print("===There is Missing value===")

    target = data["target"]
    data.drop("target", axis=1, inplace=True)

    # dataset split, 80% training 20 validation
    n_train = int(data.shape[0] * 0.8)
    train_X = data[:n_train].to_numpy()
    val_X = data[n_train:].to_numpy()
    train_y = target[:n_train].to_numpy()
    val_y = target[n_train:].to_numpy()

    train_X = train_X.astype(np.float32)
    val_X = val_X.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_y = val_y.astype(np.float32)

    # Data preprocessing
    scaler = StandardScaler().fit(train_X)
    train_X -= scaler.mean_
    val_X -= scaler.mean_

    return train_X, val_X, train_y, val_y


if __name__ == '__main__':
    train_X, val_X, train_y, val_y = load_data('uniform_synthetic_data_Butane_N=5000_D=12 - T2.xlsx')
    np.savez('BLEVE_simulated_open',train_X, val_X, train_y, val_y)