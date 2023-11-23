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


def compute_auc(data_x, data_y, params, strat):

    if strat:
        X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, stratify=data_y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

    bst = XGBClassifier(booster='gbtree', max_depth=params['max_depth'], n_estimators=params['n_estimators'],
                        learning_rate=params['learning_rate'], reg_lambda=params['reg_lambda'], gamma=params['gamma'],
                        objective='binary:logistic')
    bst.fit(X_train, y_train)

    pred_train = bst.predict_proba(X_train)
    pred_train = pred_train[:,1]
    pred_val = bst.predict_proba(X_val)
    pred_val = pred_val[:,1]

    train_auc = roc_auc_score(y_train, pred_train)
    val_auc = roc_auc_score(y_val, pred_val)
    return train_auc, val_auc


def main(category, suffix, param_set, stratify):

    # fixed params
    num_cross = 20

    use_nni_params = True

    if use_nni_params:
        with open('best_params_xgb.json') as json_file:
            params = json.load(json_file)[f'nni_opti_param_{category}_{param_set}']
    else:
        params = {"max_depth": 7,
                  "n_estimators": 100,
                  "learning_rate": 0.01,
                  "reg_lambda": 10,
                  "gamma": 0.5}

    file_input = f"Input_train_{suffix}.csv"
    file_label = f"Output_train.csv"

    set_seed(0)
    data_x, data_y = load_data(file_input, file_label, category)
    train_auc = []
    val_auc = []

    for i in range(num_cross):
        temp_train_auc, temp_val_auc = compute_auc(data_x, data_y, params, stratify)
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

    print("Valid mean is", np.mean(val_auc))
    print("Valid std is", np.std(val_auc))
    print("Train mean is", np.mean(train_auc))
    print("Train std is", np.std(train_auc))

    return np.mean(val_auc), np.mean(train_auc), np.std(val_auc), np.std(train_auc)

if __name__ == "__main__":

    category = 'GVH2'
    suffix_list = ['Basic_dx', 'HLA', 'KIR', 'Basic_dx_HLA', 'Basic_dx_KIR', 'HLA_KIR', 'Basic_dx_HLA_sum',
                   'Basic_dx_KIR_sum', 'Basic_dx_HLA_KIR', 'Basic_dx_HLA_KIR_sum', 'Basic_dx_HLA_KIR_sum_sum']
    param_set_list = ['Age', 'HLA', 'Age_HLA', 'Age_HLA_sum']
    stratify_list = [False, True]
    results = []
    count = 0

    for strat in stratify_list:
        for par_set in param_set_list:
            for suff in suffix_list:
                print('')
                print('count is:', count)
                results.append(main(category, suffix, param_set, stratify))
                count += 1

    save = np.asmatrix(results)
    np.savetxt(f"grid_xgb_{category}_dx_all.csv", save, delimiter=",")