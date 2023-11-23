import pandas as pd

from data_functions import get_data
import data_functions
import preProcessing
from sklearn.model_selection import train_test_split
import embedding_model
import my_model
import akiva_model
EMB_FLAG = False
params = {"batch_size": 80,
          "MAX_EPOCHS": 70,
          "hidden1": 50,
          "hidden2": 100,
          "dropout": 0.8,
          "weight_decay": 0.001,
          "lr": 0.001,
          "alpha": 0.2}



if __name__ == '__main__':
    # data_features, labels = data_functions.get2301data()
    import process_8dec22_data
    data_features, labels = process_8dec22_data.get_akiva_data()

    # data_features, labels = data_functions.get_new_data()
    # data_features, labels = get_data(embedding_flag=EMB_FLAG)
    # data_features = data_functions.get_AKIVA_input()
    indexes = data_features.index
    # data_features = preProcessing.features_pre_process(data_features)
    # labels = preProcessing.labels_pre_process(labels)
    # if EMB_FLAG:
    #     emb_features = data_functions.get_embedding_features()
    #     commom = emb_features.index.intersection(indexes)
    #     data_features = pd.concat((data_features.loc[commom], emb_features.loc[commom]), axis=1)
    #     labels = labels.loc[commom]

    # Train test split
    X_trainind, X_testind, y_train, y_test = train_test_split(
    data_features[0].index, labels, test_size=0.1, random_state=42)

    X_trainind, X_valind, y_train, y_val = train_test_split(
    X_trainind, y_train, test_size=0.2, random_state=42)

    X_train = [X.loc[X_trainind] for X in data_features]
    X_val = [X.loc[X_valind] for X in data_features]
    X_test = [X.loc[X_testind] for X in data_features]

    y_train = preProcessing.labels_pre_process(y_train)
    y_val = preProcessing.labels_pre_process(y_val)
    y_test = preProcessing.labels_pre_process(y_test)

    # Train
    # embedding_model.train_model(X_train, y_train, X_val, y_val, params)
    # my_model.train_model(X_train[0].values, y_train, X_val[0].values, y_val, params)
    akiva_model.train_model(X_train[0].values, y_train, X_val[0].values, y_val)
