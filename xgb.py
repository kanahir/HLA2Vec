from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import random
import json

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def load_data(file_x, file_y, category):

    pre_data_x = pd.read_csv(file_x, delimiter=',')
    data_x = pre_data_x.values

    pre_data_y = pd.read_csv(file_y, delimiter=',')
    data_y_bin = pre_data_y[f'Del{category}'].values
    data_flag = pre_data_y[f'Flag{category}'].values

    data_x = np.array([data_x[i] for i in range(len(data_flag)) if data_flag[i] == 1])
    data_y_bin = np.array([data_y_bin[i] for i in range(len(data_flag)) if data_flag[i] == 1])

    return data_x, data_y_bin


def compute_auc(X_train, X_val, y_train, y_val, params):


    bst = XGBClassifier(booster='gbtree', max_depth=params['max_depth'], n_estimators=params['n_estimators'],
                        learning_rate=params['learning_rate'], reg_lambda=params['reg_lambda'], gamma=params['gamma'],
                        objective='binary:logistic', eval_metric="logloss")
    bst.fit(X_train, y_train)

    pred_train = bst.predict_proba(X_train)
    pred_train = pred_train[:,1]
    pred_val = bst.predict_proba(X_val)
    pred_val = pred_val[:,1]

    train_auc = roc_auc_score(y_train, pred_train)
    val_auc = roc_auc_score(y_val, pred_val)
    return train_auc, val_auc


def main(X_train, y_train, X_val, y_val, stratify=False):
    auc_val_vec = {}
    auc_train_vec = {}
    for i, task in enumerate(["Sur", "GVH2", "GVH3", "CGVH", "DFS"]):
        print(task)
        y_t = y_train[i]
        y_v = y_val[i]
        # Flag_train = y_t["Flag" + task]
        Flag_train = y_t[task + "_flag"]

        labels_train = y_t["Del" + task]
        # Flag_val = y_v["Flag" + task]
        Flag_val = y_v[task + "_flag"]

        labels_val = y_v["Del" + task]

        X_train_for_task = X_train.loc[Flag_train != 0].values
        labels_train_for_task = labels_train.loc[Flag_train != 0]
        labels_val_for_task = labels_val.loc[Flag_val != 0]
        X_val_for_task = X_val.loc[Flag_val != 0].values
        # fixed params
        # num_cross = 20
        num_cross = 1

        params = {"max_depth": 7,
                  "n_estimators": 100,
                  "learning_rate": 0.01,
                  "reg_lambda": 10,
                  "gamma": 0.5}

        # file_input = f"Input_train_Basic_dx_HLA_KIR.csv"
        # file_label = f"Output_train.csv"
        #
        # set_seed(0)
        # data_x, data_y = load_data(file_input, file_label, category)
        train_auc = []
        val_auc = []

        for i in range(num_cross):
            temp_train_auc, temp_val_auc = compute_auc(X_train_for_task, X_val_for_task, labels_train_for_task, labels_val_for_task, params)
            train_auc.append(temp_train_auc)
            val_auc.append(temp_val_auc)

        # print(category)
        # print('')
        # print("results_train")
        # for t in train_auc:
        #     print(t)
        #
        # print('')
        # print("results_valid")
        # for v in val_auc:
        #     print(v)

        print(f"Valid mean is {np.mean(val_auc) :.2f}")
        print(f"Valid std is {np.std(val_auc) :.2f}")
        print(f"Train mean is {np.mean(train_auc) :.2f}")
        print(f"Train std is {np.std(train_auc) :.2f}")
        np.std(val_auc), np.std(train_auc)
        auc_val_vec[task] = np.mean(val_auc)
        auc_train_vec[task] = np.mean(train_auc)

    return auc_val_vec, auc_train_vec

if __name__ == "__main__":

    category = 'GVH2'
    suffix_list = ['Basic_dx', 'HLA', 'KIR', 'Basic_dx_HLA', 'Basic_dx_KIR', 'HLA_KIR', 'Basic_dx_HLA_sum',
                   'Basic_dx_KIR_sum', 'Basic_dx_HLA_KIR', 'Basic_dx_HLA_KIR_sum', 'Basic_dx_HLA_KIR_sum_sum']
    param_set_list = ['Age', 'HLA', 'Age_HLA', 'Age_HLA_sum']
    # stratify_list = [False, True]
    stratify_list= [False]
    results = []
    count = 0

    for strat in stratify_list:
        for par_set in param_set_list:
            for suff in suffix_list:
                print('')
                print('count is:', count)
                results.append(main(category, stratify=True))
                count += 1

    save = np.asmatrix(results)
    np.savetxt(f"grid_xgb_{category}_dx_all.csv", save, delimiter=",")