import pandas as pd
import numpy as np

import main_script
from sklearn.model_selection import train_test_split
import preProcessing
import warnings
from sklearn import preprocessing
import pickle
from sksurv.linear_model import CoxPHSurvivalAnalysis
import copy
from sksurv.compare import compare_survival
import akiva_model

path_to_dir = "/home/kanna/PycharmProjects/Gen2Vec/"
categories = ['GVH2', 'GVH3', 'CGVH', 'DFS', 'Sur']

def get_score():
    # check if the data is already saved
    try:
        X_test_copy = pd.read_pickle("X_test_score_demographic_without_processing.pkl")

    except:
        X_train, y_train, X_test, y_test = preProcessing.get_processed_train_and_test()
        # find the columns in X_train that are not in X_test

        auc_val_vec2, auc_train_vec2, correlation_vec, models = akiva_model.train_model(X_train.values, y_train, X_test.values,
                                                                                y_test)
        X_test_copy = X_test.copy()
        X_train = X_train.__array__()
        X_test = X_test.__array__()
        scaler = preprocessing.StandardScaler().fit(X_train[:, :2])
        X_train[:, :2] = scaler.transform(X_train[:, :2])
        X_test[:, :2] = scaler.transform(X_test[:, :2])
        for label, model in models.items():
            X_test_copy[f"{label}_score"] = np.max(model.fit_transform(X_test).detach().numpy(), axis=1)
        # union with the labels
        for y in y_test:
            X_test_copy = pd.concat([X_test_copy, y], axis=1)
        # union X test with all the inpormation
        # X_test = preProcessing.get_all_test()
        # concat X_test with X_test_copy and drop common columns
        unique_cols = list(set(X_test.columns) - set(X_test_copy.columns))
        X_test = X_test[unique_cols]
        score_cols = [label + "_score" for label in models.keys()]
        X_test_copy_all = pd.concat([X_test, X_test_copy[score_cols]], axis=1)
        for y in y_test:
            X_test_copy_all = pd.concat([X_test_copy_all, y], axis=1)

        # save the data
        with open("X_test_score_demographic_without_processing.pkl", "wb") as f:
            pickle.dump(X_test_copy_all, f)
    return X_test_copy

def split_by_score(X, cat):
    X["score"] = pd.qcut(X[f"{cat}_score"], 2, labels=False)
    return X

def multivariable():
    significant_hla = pd.DataFrame(columns=['GVH2 Feature Coeff', 'GVH2 Score Coeff', 'GVH2 P value Feature', 'GVH2 P value Score',
                                        'GVH3 Feature Coeff', 'GVH3 Score Coeff', 'GVH3 P value Feature', 'GVH3 P value Score',
                                        'CGVH Feature Coeff', 'CGVH Score Coeff', 'CGVH P value Feature', 'CGVH P value Score',
                                        'DFS Feature Coeff', 'DFS Score Coeff', 'DFS P value Feature','DFS P value Score',
                                        'Sur Feature Coeff', 'Sur Score Coeff', 'Sur P value Feature', 'Sur P value Score'])
    significant_kir = significant_hla.copy()

    for i, cat in enumerate(categories):
        X = get_score()
        # count mismatches
        cols_to_sum = [col for col in X.columns if len(col.split("-")) == 2 and col.split("-")[1] == '1']
        X["mismatch"] = X[cols_to_sum].sum(axis=1)
        # X.drop(cols_to_sum, axis=1, inplace=True)
        data = split_by_score(X, cat)
        features_types = ["Demographics", "HLA", "KIR"]

        features_to_remove = ["DaySur", "Sur_flag", "DayGVH2", "GVH2_flag", "DayGVH3", "GVH3_flag",
                              "DayCGVH", "CGVH_flag",
                              "DayDFS", "DFS_flag"]  # "intxmscgvhd"

        label_interval = ["Day" + cat, cat + "_flag"]
        #
        data_heder = pd.read_csv(f"{path_to_dir}All_Data.csv", header=1)


        heder = data_heder.columns
        specipic_names = data.columns
        days = label_interval[0]
        label = label_interval[1]
        task_name = days[3:]

        # for i in range(len(heder)):
        #     # print(f"{heder[i]},{specipic_names[i]}")
        #     if not any([ftype in heder[i] for ftype in features_types]):
        #         features_to_remove.append(specipic_names[i])
        #
        # features_to_remove_label = copy.deepcopy(features_to_remove)  # "intxmscgvhd"
        # for f in label_interval:
        #     while f in features_to_remove_label:
        #         features_to_remove_label.remove(f)
        #     data[f] = data[f].apply(lambda x: x if (x != -1) else np.nan)
        #     data = data[data[f].notna()]
        # data = data.drop(features_to_remove_label, axis=1)  # 'pseudoid',
        # data = data.drop(features_to_remove, axis=1)
        order = [i for i in range(len(data.columns))]
        order[0] = 1
        order[1] = 0
        data = data.iloc[:, order]

        data[label] = data[label].apply(lambda x: True if x == 1.0 else False)
        y_data = data.loc[:, [label, days]]
        # y_data["event 1y"] = list_event_after_year
        # take x_data without the label and the days
        x_data = data.drop([label, days], axis=1)
        feature_pvalue_dict = {}
        x_data = preProcessing.replace_HLA_akiva_data(x_data)
        x_data = preProcessing.remove_sparse_columns(x_data)
        y_data = [tuple(x) for x in y_data.to_numpy()]
        dt = np.dtype('bool,float')
        y_data = np.asarray(y_data, dt)
        # check if there is a nan or inf in the y_data
        columns_HLA = [data_heder.loc[0, col] for col in data_heder.columns if
                       'HLA' in col and data_heder.loc[0, col] in x_data.columns] + ["mismatch"]

        columns_KIR = [data_heder.loc[0, col] for col in data_heder.columns if
                       'KIR' in col and data_heder.loc[0, col] in x_data.columns]
        columns = columns_HLA + columns_KIR
        columns = ["DRB1-3"]
        for col in columns:
            col_back = x_data[col].copy()
            if col == "score" or np.any([category in col for category in categories]):
                continue
            if x_data[col].unique().size > 2:

                col_median = x_data[col].median()
                # replace the values with 0 or 1
                group1 = x_data[x_data[col] > col_median]
                if group1.shape[0] < 10:
                    x_data[col] = [1 if float(x) >= col_median else 0 for x in x_data[col]]
                else:
                    x_data[col] = [1 if float(x) > col_median else 0 for x in x_data[col]]


            x_for_check = x_data.loc[:, [col, "score"]]
            try:
                chisq, pvalue_feature, stats, covar = compare_survival(
                    y_data, x_for_check[col], return_stats=True)
                chisq, pvalue_score, stats, covar = compare_survival(
                    y_data, x_for_check["score"], return_stats=True)
                print(f"{cat} pvalue feature: {pvalue_feature}, pvalue score: {pvalue_score}")
                # x_for_check.to_csv(f"{path_to_dir}_x_DEB1-3.csv")
                # y_data.to_csv(f"{path_to_dir}_y_DEB1-3.csv")
                if pvalue_feature == 0 or pvalue_score == 0:
                    a=1
                rsf = CoxPHSurvivalAnalysis().fit(x_for_check, y_data)
                if col in columns_HLA:
                    significant_hla.loc[col, f'{cat} Feature Coeff'] = rsf.coef_[0]
                    significant_hla.loc[col, f'{cat} Score Coeff'] = rsf.coef_[1]
                    significant_hla.loc[col, f'{cat} P value Feature'] = pvalue_feature
                    significant_hla.loc[col, f'{cat} P value Score'] = pvalue_score

                elif col in columns_KIR:
                    significant_kir.loc[col, f'{cat} Feature Coeff'] = rsf.coef_[0]
                    significant_kir.loc[col, f'{cat} Score Coeff'] = rsf.coef_[1]
                    significant_kir.loc[col, f'{cat} P value Feature'] = pvalue_feature
                    significant_kir.loc[col, f'{cat} P value Score'] = pvalue_score
            except:
                print(f"problem with {col}")

        significant_hla.astype("float").to_csv("significant_hla3.csv", float_format='%.3f')
        significant_kir.astype("float").to_csv("significant_kir3.csv", float_format='%.3f')
    a=1


if __name__ == '__main__':
    multivariable()