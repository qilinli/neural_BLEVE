import pandas as pd
import numpy as np
from random import shuffle
from sklearn.preprocessing import StandardScaler


def load_data(file):
    # Load the excel and names for all sheets
    xls = pd.ExcelFile(file, engine='openpyxl')
    sheets = xls.sheet_names
    
    # Read sheets of interest
    df = []
    for sheet in sheets:
        if "neg" in sheet:
            print(sheet)
            df.append(pd.read_excel(xls, sheet))        
    df = pd.concat(df)
    
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
    
    data_dict = {}
    data_dict["train_X"] = train_X.astype(np.float32)
    data_dict["train_y"] = train_y.astype(np.float32)
    data_dict["val_X"] = val_X.astype(np.float32)
    data_dict["val_y"] = val_y.astype(np.float32)
    data_dict["test_X"] = test_X.astype(np.float32)
    data_dict["test_y"] = test_y.astype(np.float32)

    # Data preprocessing
    scaler = StandardScaler().fit(train_X)
    data_dict["mean"] = scaler.mean_
    data_dict["std"] = scaler.scale_

    print("The size of training data: ", train_X.shape)
    print("The size of training data: ", val_X.shape)
    print("The size of training data: ", test_X.shape)
    print("mean", data_dict["mean"])
    print("std", data_dict["std"])
    
    return data_dict


if __name__ == '__main__':
    data_dir = "./data/"
    data_dict = load_data(data_dir + 'Peak pressure time FLACS.xlsx')
    np.save(data_dir + 'peak_pressure_time_neg', data_dict)