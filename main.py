import pandas as pd
import data_functions
import preProcessing
from sklearn.model_selection import train_test_split
from Models import embedding_model, embedding_model_separate, base_model, separate_model

EMB_FLAG = True

params = {"batch_size": 80,
          "MAX_EPOCHS": 70,
          "hidden1": 50,
          "hidden2": 100,
          "dropout": 0.8,
          "weight_decay": 0.001,
          "lr": 0.001,
          "alpha": 0.2}


if __name__ == '__main__':

    # data_features, labels = data_functions.get_new_data()
    data_features, labels = preProcessing.get_data(embedding_flag=EMB_FLAG)
    # data_features = data_functions.get_AKIVA_input()
    indexes = data_features.index
    data_features = preProcessing.features_pre_process(data_features)
    labels = preProcessing.labels_pre_process(labels)
    if EMB_FLAG:
        emb_features = data_functions.get_embedding_features()
        commom = emb_features.index.intersection(indexes)
        data_features = pd.concat((data_features.loc[commom], emb_features.loc[commom]), axis=1)
        labels = labels.loc[commom]

    # Train test split
    X_train, y_train, X_test, y_test = train_test_split(data_features, labels, test_size=0.1, random_state=42)
    X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    y_train = preProcessing.labels_pre_process(y_train)
    y_val = preProcessing.labels_pre_process(y_val)
    y_test = preProcessing.labels_pre_process(y_test)

    # Train
    embedding_model.train_model(X_train, y_train, X_val, y_val, params)
    # model.train_model(X_train[0].values, y_train, X_val[0].values, y_val, params)
    # separate_model.train_model(X_train[0].values, y_train, X_val[0].values, y_val)
