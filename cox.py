from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def predict_score(X_train, y_train, X_val, y_val):
    auc_val_vec = {}
    auc_train_vec = {}
    for y_train_per_task, y_val_per_task in zip(y_train, y_val):
        task_name = y_train_per_task.columns[0][3:]

        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        X_train0 = add_constant(X_train)
        vif = pd.Series([variance_inflation_factor(X_train0.values, i)
                         for i in range(X_train0.shape[1])],
                        index=X_train0.columns)
        # take only features with vif < 5
        # X_train_dropped = X_train0.loc[:, vif < 5]
        # X_val_dropped = X_val.loc[:, vif < 5]

        # remove last column of each feature
        # columns_to_remove = [col for col in X_train.columns if col.endswith('_1')]
        # X_train_dropped = X_train.drop(columns=columns_to_remove)
        # X_val_dropped = X_val.drop(columns=columns_to_remove)
        #
        # # drop columns with all ones
        # X_train_dropped = X_train_dropped.loc[:, (X_train_dropped != 1).sum(axis=0) > 50]
        # X_val_dropped = X_val_dropped.loc[:, X_train_dropped.columns]
        #
        # # data preparation
        X_train_per_task = pd.concat([X_train_dropped, y_train_per_task], axis=1)
        # y_train_per_task = y_train_per_task.loc[X_train_per_task[task_name + '_flag'] == 1]
        # X_train_per_task = X_train_per_task.loc[X_train_per_task[task_name + '_flag'] == 1]
        X_train_per_task = X_train_per_task.drop(columns=[task_name + '_flag'])
        #
        X_val_per_task = pd.concat([X_val_dropped, y_val_per_task], axis=1)
        # y_val_per_task = y_val_per_task.loc[X_val_per_task[task_name + '_flag'] == 1]
        # X_val_per_task = X_val_per_task.loc[X_val_per_task[task_name + '_flag'] == 1]
        # X_val_per_task = X_val_per_task.drop(columns=[task_name + '_flag'])


        cph = CoxPHFitter()
        cph.fit(X_train_per_task, duration_col='Day' + task_name, event_col='Del' + task_name)
        y_train_predict = cph.predict_survival_function(X_train_per_task, times=[365])
        pred_val = cph.predict_survival_function(X_val_per_task, times=[365])

        y_train_true_class = y_train_per_task['Del' + task_name].values
        y_train_predict_class = y_train_predict.values

        y_val_true_class = y_val_per_task['Del' + task_name].values
        y_val_predict_class = pred_val.values

        auc_score_train = roc_auc_score(y_train_true_class.reshape(-1, 1), y_train_predict_class.reshape(-1, 1))
        auc_score_val = roc_auc_score(y_val_true_class.reshape(-1, 1), y_val_predict_class.reshape(-1, 1))

        auc_val_vec[task_name] = auc_score_val
        auc_train_vec[task_name] = auc_score_train



    cph.fit(X_train_per_task, duration_col='Day' + task_name, event_col='Del' + task_name)

    return auc_val_vec, auc_train_vec


if __name__ == '__main__':
    rossi = load_rossi()

    cph = CoxPHFitter()
    cph.fit(rossi, duration_col='week', event_col='arrest')

    cph.print_summary()
    X = rossi

    cph.predict_survival_function(X)
    cph.predict_median(X)
    cph.predict_partial_hazard(X)
    a = 1

    # all regression models can be used here, WeibullAFTFitter is used for illustration
    wf = CoxPHFitter().fit(rossi, "week", "arrest")

    # filter down to just censored subjects to predict remaining survival
    censored_subjects = rossi.loc[~rossi['arrest'].astype(bool)]
    censored_subjects_last_obs = censored_subjects['week']

    # predict new survival function
    a = wf.predict_survival_function(censored_subjects, conditional_after=censored_subjects_last_obs)
    wf.predict_survival_function(censored_subjects).plot()