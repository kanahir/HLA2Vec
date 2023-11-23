import copy
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import random
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.compare import compare_survival

from lifelines import CoxPHFitter

import statistics
from sklearn import neighbors
import operator
import pathlib
from collections import defaultdict
import main_script
import seaborn as sns
import pickle
random.seed(0)
import preProcessing

path_to_dir = "/home/kanna/PycharmProjects/Gen2Vec/"
def calc_auc(surv, y_train, year):
    list_target = []
    list_output = []
    for i, fn in enumerate(surv):
        surv_array = fn.a * fn.y + fn.b * fn.x
        event = y_train[i][0]
        intex = y_train[i][1]
        year = min(year, len(surv_array) - 5)

        if not event and intex >= year:
            list_target.append(0)
            list_output.append(surv_array[year - 1])
        elif event and intex < year:
            list_target.append(1)
            list_output.append(surv_array[year - 1])
        elif event and intex >= year:
            list_target.append(0)
            list_output.append(surv_array[year - 1])

    fpr, tpr, _ = metrics.roc_curve(np.array(list_target), np.array(list_output))
    roc_auc_train = metrics.auc(fpr, tpr)
    return roc_auc_train

def calc_correlation(surv, y_train, year):
    list_target = []
    list_output = []
    for i, fn in enumerate(surv):
        surv_array = fn.a * fn.y + fn.b * fn.x
        event = y_train[i][0]
        intex = y_train[i][1]
        year = min(year, len(surv_array) - 5)

        if not event and intex >= year:
            list_target.append(0)
            list_output.append(surv_array[year - 1])
        elif event and intex < year:
            list_target.append(1)
            list_output.append(surv_array[year - 1])
        elif event and intex >= year:
            list_target.append(0)
            list_output.append(surv_array[year - 1])

    corr = stats.spearmanr(np.array(list_target), np.array(list_output))
    return corr[0]

#####multi_prodtype
def frange(start, stop, step=1.0):
    i = start
    while i < stop:
        yield i
        i += step


def plot_confusion_matrix(labels, cm, classifier_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, interpolation='nearest')
    # plt.matshow(cm)
    plt.title('Confusion matrix- ' + classifier_name)  # ,classifier_name
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    classifier_name = classifier_name.replace('.', '').replace(':', '_')
    plt.savefig("plots/confusion_matrix_" + classifier_name)
    plt.clf()
    plt.close('all')


# def calc_cox(X_train, y_train, X_val, y_val):
#     task_name = y_train.columns[0][3:]
#     X_train = pd.concat([X_train, y_train], axis=1)
#     X_val = pd.concat([X_val, y_val], axis=1)
#
#     year = 365
#     list_test = defaultdict(list)
#     for label_interval in [["DayGVH2", "FlagGVH2"], ["DayGVH3", "FlagGVH3"], ["DayCGVH", "FlagCGVH"],
#                            ["DayDFS", "FlagDFS"],
#                            ["DaySur", "FlagSur"]]:  #
#         days = "Day" + task_name
#         label = "Flag" + task_name
#         data_heder = pd.read_csv("All_Data.csv", header=1)
#         data = pd.read_csv("All_Data.csv", header=2)
#
#         heder = data_heder.columns
#         specipic_names = data.columns
#         for i in range(len(heder)):
#             # print(f"{heder[i]},{specipic_names[i]}")
#             if 'Del' in specipic_names[i] or 'KIR' in heder[i] or "HLA" in heder[i]:  # todo - choose what to remove
#                 features_to_remove.append(specipic_names[i])
#
#         """column = "Age donor"
#         data[column] = data[column].astype(float)
#         mean = data[column].mean()
#         data[column] = data[column].fillna(mean)"""
#         # year = 365 if label != "FlagGVH2" else 210
#
#         features_to_remove_label = copy.deepcopy(features_to_remove)  # "intxmscgvhd"
#         for f in label_interval:
#             features_to_remove_label.remove(f)
#             data[f] = data[f].apply(lambda x: x if (x != -1) else np.nan)
#             data = data[data[f].notna()]
#         data = data.drop(features_to_remove_label, axis=1)  # 'pseudoid',
#         order = [i for i in range(len(data.columns))]
#         order[0] = 1
#         order[1] = 0
#         data = data.iloc[:, order]
#         """list_event_after_year = []
#         for idx, row in data.iterrows():
#             event = row[label]
#             intx = row[days]
#             if event == 1 and intx > 365:
#                 list_event_after_year.append(0)
#             else:
#                 list_event_after_year.append(event)"""
#         data[label] = data[label].apply(lambda x: True if x == 1.0 else False)
#         y_data = data.iloc[:, :2]
#         # y_data["event 1y"] = list_event_after_year
#         x_data = data.iloc[:, 2:]
#         X_train_all, X_test, y_train_all, y_test = model_selection.train_test_split(x_data, y_data,
#                                                                                     test_size=0.2)  # , shuffle=False)
#
#         y_test = [tuple(x) for x in y_test.to_numpy()]
#         dt = np.dtype('bool,float')
#         y_test = np.asarray(y_test, dt)
#
#         # normalize
#         """std = x_train_data.std()
#         x_valid_data = (x_valid_data - x_train_data.mean()) / x_train_data.std()
#         x_test_data = (x_test_data - x_train_data.mean()) / x_train_data.std()
#         x_train_data = (x_train_data - x_train_data.mean()) / x_train_data.std()"""
#
#         # exit()
#         file_res = open(folder_to_results + 'auc_res_csa_' + str(label) + str(int(year / 12)) + 'y' + '.csv', 'w')
#         file_res.write('l1,auc_test\n')
#         # random forest
#         # print('csa ' + label + str(int(year / 12)) + 'y')
#         l1_list = [0.01, 0.1, 0.5, 0.8]
#         max_auc = 0
#
#         dict_valid_line = {}
#         for l1 in l1_list:
#             auc_valid_list = []
#
#             count_y = 0
#             count_n = 0
#             algo_name = 'cox- ' + label
#             params = 'depth-' + str(l1)
#             # print(params)
#
#             for k in range(5):
#                 X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train_all, y_train_all,
#                                                                                       test_size=0.2)
#
#                 y_train = [tuple(x) for x in y_train.to_numpy()]
#                 dt = np.dtype('bool,float')
#                 y_train = np.asarray(y_train, dt)
#
#                 y_valid = [tuple(x) for x in y_valid.to_numpy()]
#                 dt = np.dtype('bool,float')
#                 y_valid = np.asarray(y_valid, dt)
#
#                 rsf = CoxPHSurvivalAnalysis(alpha=l1).fit(X_train, y_train)
#
#                 ##train auc
#                 surv = rsf.predict_cumulative_hazard_function(X_train)
#                 roc_auc_train = calc_auc(surv, y_train, year)
#
#                 ##valid auc
#                 surv = rsf.predict_cumulative_hazard_function(X_valid)
#                 roc_auc_valid = calc_auc(surv, y_valid, year)
#                 auc_valid_list.append(roc_auc_valid)
#
#             roc_auc_valid = sum(auc_valid_list) / 5
#
#             ##test auc
#             y_train = [tuple(x) for x in y_train_all.to_numpy()]
#             dt = np.dtype('bool,float')
#             y_train = np.asarray(y_train, dt)
#             rsf = CoxPHSurvivalAnalysis(alpha=l1).fit(X_train_all, y_train)
#
#             ##train auc
#             surv = rsf.predict_cumulative_hazard_function(X_train_all)
#             roc_auc_train = calc_auc(surv, y_train, year)
#
#             surv = rsf.predict_cumulative_hazard_function(X_test)
#             roc_auc = calc_auc(surv, y_test, year)
#             # print(f'{label} & {round(roc_auc_train,3)} & {round(roc_auc_valid,3)} \pm {round(statistics.pstdev(auc_valid_list),3)} & {round(roc_auc,3)} \\\\ {params}')
#             # dict_valid_line[round(roc_auc_valid,3)] = f'{label} & {round(roc_auc_train,3)} & {round(roc_auc_valid,3)} \pm {round(statistics.pstdev(auc_valid_list),3)} & {round(roc_auc,3)} \\\\'
#             # dict_valid_line[f'{label} & {round(roc_auc_train,3)} & $ {round(roc_auc_valid,3)} \pm {round(statistics.pstdev(auc_valid_list),3)} $ & {round(roc_auc,3)} \\\\'] = round(roc_auc_valid,3)
#             dict_valid_line[str(round(roc_auc, 3))] = round(roc_auc_valid, 3)
#
#             # print(str(count_y) + ' ' + str(count_n)+ " score:" +  str(rsf.score(X_test, y_test)) )
#
#             max_auc = max(roc_auc, max_auc)
#
#             file_res.write(str(l1) + str(roc_auc) + '\n')
#
#             # plt.step(rsf.event_times_, s, where="post", label=str(i))
#
#         # max_auc = max(auc, max_auc)
#
#         # print( max(dict_valid_line.items(), key=operator.itemgetter(1))[0])
#         list_test[label].append(max(dict_valid_line.items(), key=operator.itemgetter(1))[0])

def plot_heatmap(df, file_name):
    # drop columns that contain nan
    df = df.dropna(axis=1, how='any')
    data_heder = pd.read_csv(f"{path_to_dir}All_Data.csv", header=1)
    columns_HLA = [data_heder.loc[0, col] for col in data_heder.columns if 'HLA' in col and data_heder.loc[0, col] in df.index]
    df_to_plot = df.loc[columns_HLA, :]
    f = sns.heatmap(df_to_plot, annot=False, cmap="coolwarm", cbar=False,  vmin=-1, vmax=1)
    f.set_xticklabels([col.split("_")[0] for col in df_to_plot.columns], rotation=0)
    plt.tight_layout()
    plt.savefig(file_name + '_HLA_nosparse.png')
    plt.show()

    columns_KIR = [data_heder.loc[0, col] for col in data_heder.columns if 'KIR' in col and data_heder.loc[0, col] in df.index]
    df_to_plot = df.loc[columns_KIR, :]
    f = sns.heatmap(df_to_plot, annot=False, cmap="coolwarm", cbar=False, vmin=-1, vmax=1)
    f.set_xtickHLlabels([col.split("_")[0] for col in df_to_plot.columns], rotation=0)
    plt.tight_layout()
    plt.savefig(file_name + '_KIR_nosparse.png')
    plt.show()

    columns_DEOGRAPICS = [data_heder.loc[0, col] for col in data_heder.columns if 'Demographics' in col and data_heder.loc[0, col] in df.index]
    df_to_plot = df.loc[columns_DEOGRAPICS, :]
    f = sns.heatmap(df_to_plot, annot=False, cmap="coolwarm", cbar=False, vmin=-1, vmax=1)
    f.set_xticklabels([col.split("_")[0] for col in df_to_plot.columns], rotation=0)
    plt.tight_layout()
    plt.savefig(file_name + '_DEOGRAPICS_KIR_nosparse.png')
    plt.show()

    a=1

def calc_cox_with_pvalue():
    test_index = pd.read_csv(f"Output_test.csv")[['ID-pair']]
    features_types = ["Demographics", "HLA", "KIR"]

    features_to_remove = ["DaySur", "FlagSur", "DayGVH2", "FlagGVH2", "DayGVH3", "FlagGVH3", "DayCGVH", "FlagCGVH",
                          "DayDFS", "FlagDFS", "ID-pair"]  # "intxmscgvhd"
    data = pd.read_csv(f"{path_to_dir}All_Data.csv", header=2)

    feature_pvalue = pd.DataFrame(index=data.columns, columns=['GVH2', 'GVH3', 'CGVH', 'DFS', 'Sur'])
    feature_coeff = pd.DataFrame(columns=['GVH2', 'GVH3', 'CGVH', 'DFS', 'Sur'])

    for label_interval in [["DayGVH2", "FlagGVH2"], ["DayGVH3", "FlagGVH3"], ["DayCGVH", "FlagCGVH"], ["DayDFS", "FlagDFS"],
                           ["DaySur", "FlagSur"]]:
        #
        data_heder = pd.read_csv(f"{path_to_dir}All_Data.csv", header=1)
        data = pd.read_csv(f"{path_to_dir}All_Data.csv", header=2)

        # delete test
        # data = data[~data['ID-pair'].isin(test_index['ID-pair'])]

        heder = data_heder.columns
        specipic_names = data.columns
        days = label_interval[0]
        label = label_interval[1]
        task_name = days[3:]

        for i in range(len(heder)):
            # print(f"{heder[i]},{specipic_names[i]}")
            if not any([ftype in heder[i] for ftype in features_types]):
                features_to_remove.append(specipic_names[i])


        features_to_remove_label = copy.deepcopy(features_to_remove)  # "intxmscgvhd"
        for f in label_interval:
            while f in features_to_remove_label:
                features_to_remove_label.remove(f)
            data[f] = data[f].apply(lambda x: x if (x != -1) else np.nan)
            data = data[data[f].notna()]
        data = data.drop(features_to_remove_label, axis=1)  # 'pseudoid',
        order = [i for i in range(len(data.columns))]
        order[0] = 1
        order[1] = 0
        data = data.iloc[:, order]

        data[label] = data[label].apply(lambda x: True if x == 1.0 else False)
        y_data = data.iloc[:, :2]
        # y_data["event 1y"] = list_event_after_year
        x_data = data.iloc[:, 2:]
        feature_pvalue_dict = {}
        x_data = preProcessing.replace_HLA_akiva_data(x_data)
        x_data = preProcessing.remove_sparse_columns(x_data)
        feature_pvalue = feature_pvalue.loc[[col for col in x_data.columns if col in feature_pvalue.index], :]
        y_data = [tuple(x) for x in y_data.to_numpy()]
        dt = np.dtype('bool,float')
        y_data = np.asarray(y_data, dt)
        rsf = CoxPHSurvivalAnalysis().fit(x_data, y_data)

        coeffs = pd.Series({x_data.columns[i]: rsf.coef_[i] for i in range(len(x_data.columns)) })
                           # if
                           #  x_data.columns[i] in coeffs_df.index})
        feature_coeff[task_name] = coeffs

        for feature in x_data.columns:
            # y_data_task = y_data.values
            # replace to binary the first column
            # y_data_task = np.array([(True, x[1]) if x[0] else (False, x[1]) for x in y_data_task], dtype="bool,f")
            feature_column = x_data.loc[:, feature]
            try:
                chisq, pvalue, stats, covar = compare_survival(
                    y_data, feature_column, return_stats=True)
                feature_pvalue.loc[feature, task_name] = pvalue
                feature_pvalue_dict[feature] = pvalue


            except:
                pass

#         plot dictionary as bar plot
        plt.figure(figsize=(20, 10))
        # set dict values to -log10
        feature_pvalue_dict = {k: -np.log10(v) for k, v in feature_pvalue_dict.items()}
        plt.bar(range(len(feature_pvalue_dict)), list(feature_pvalue_dict.values()), align='center')
        plt.xticks(range(len(feature_pvalue_dict)), list(feature_pvalue_dict.keys()), rotation=90)
        plt.ylabel('-log10 P Value')
        plt.title(f"pvalue for {task_name}")
        plt.savefig(f"pvalue for {task_name}.png")
        plt.show()
    # round 2 digits
    feature_pvalue = feature_pvalue.round(2)
    feature_coeff = feature_coeff.round(2)
    feature_pvalue.to_csv("pvalue_cox2.csv", float_format="%.2f")
    feature_coeff.to_csv("coeff_cox2.csv", float_format="%.2f")
    return feature_pvalue




def calc_coeff():
    test_index = pd.read_csv(f"Output_test.csv")[['ID-pair']]
    features_types = ["Demographics", "HLA", "KIR"]
    # features_types = ["Demographics"]

    # Create output directory if it doesn't exist
    # pathlib.Path("plots").mkdir(parents=False, exist_ok=True)
    # folder_to_results = 'coxPHS/'
    # pathlib.Path(folder_to_results[:-1]).mkdir(parents=False, exist_ok=True)
    features_to_remove = ["DaySur", "FlagSur", "DayGVH2", "FlagGVH2", "DayGVH3", "FlagGVH3", "DayCGVH", "FlagCGVH",
                          "DayDFS", "FlagDFS", "ID-pair"]  # "intxmscgvhd"

    year = 365
    # types_to_take_vec = [
    #     features_types[0], features_types[0:2], features_types]
    types_to_take_vec = [features_types[0]]
    cox_dict = defaultdict(dict)
    cox_dict_corr = defaultdict(dict)
    data_heder = pd.read_csv(f"{path_to_dir}All_Data.csv", header=1)
    data = pd.read_csv(f"{path_to_dir}All_Data.csv", header=2)
    data = preProcessing.replace_HLA_all_data(data_heder, data)
    data = preProcessing.remove_sparse_columns(data)
    heder = data_heder.columns
    specipic_names = data.columns
    for ind, types_to_take in enumerate(types_to_take_vec):
        auc_val_cox = {"Sur": [], "GVH2": [], "GVH3": [], "CGVH": [], "DFS": []}
        corr_val_cox = {"Sur": [], "GVH2": [], "GVH3": [], "CGVH": [], "DFS": []}
        coeffs_df = pd.DataFrame(index=[col for col in data.columns if
                                        "ID" not in col and "Del" not in col and "Day" not in col and "Flag" not in col],
                                 columns=["GVH3", "GVH2", "CGVH", "DFS", "Sur"])
        results_df = pd.DataFrame(index=["Sur_mean", "Sur_std", "GVH2_mean", "GVH2_std", "GVH3_mean",
                                         "GVH3_std", "CGVH_mean", "CGVH_std", "DFS_mean", "DFS_std"],
                                  columns=[types_to_take])

        for label_interval in [["DayGVH2", "FlagGVH2"], ["DayGVH3", "FlagGVH3"], ["DayCGVH", "FlagCGVH"],
                               ["DayDFS", "FlagDFS"],
                               ["DaySur", "FlagSur"]]:
            #
            data_heder = pd.read_csv(f"{path_to_dir}All_Data.csv", header=1)
            data = pd.read_csv(f"{path_to_dir}All_Data.csv", header=2)
            # delete test
            data = data[~data['ID-pair'].isin(test_index['ID-pair'])]

            heder = data_heder.columns
            specipic_names = data.columns
            days = label_interval[0]
            label = label_interval[1]
            task_name = days[3:]

            features_to_take = []
            for i in range(len(heder)):
                # print(f"{heder[i]},{specipic_names[i]}")
                if not any([ftype in heder[i] for ftype in features_types]):
                    features_to_remove.append(specipic_names[i])
                # if 'Del' in specipic_names[i] or 'KIR' in heder[i] or "HLA" in heder[i]:  # todo - choose what to remove
                #     features_to_take = [col for col in df if features_types_dict[col] in types_to_take]

            """column = "Age donor"
            data[column] = data[column].astype(float)
            mean = data[column].mean()
            data[column] = data[column].fillna(mean)"""
            # year = 365 if label != "FlagGVH2" else 210

            features_to_remove_label = copy.deepcopy(features_to_remove)  # "intxmscgvhd"
            for f in label_interval:
                while f in features_to_remove_label:
                    features_to_remove_label.remove(f)
                data[f] = data[f].apply(lambda x: x if (x != -1) else np.nan)
                data = data[data[f].notna()]
            data = data.drop(features_to_remove_label, axis=1)  # 'pseudoid',
            order = [i for i in range(len(data.columns))]
            order[0] = 1
            order[1] = 0
            data = data.iloc[:, order]
            """list_event_after_year = []
            for idx, row in data.iterrows():
                event = row[label]
                intx = row[days]
                if event == 1 and intx > 365:
                    list_event_after_year.append(0)
                else:
                    list_event_after_year.append(event)"""
            data[label] = data[label].apply(lambda x: True if x == 1.0 else False)
            y_data = data.iloc[:, :2]
            # y_data["event 1y"] = list_event_after_year
            x_data = data.iloc[:, 2:]
            X_train_all, X_test, y_train_all, y_test = model_selection.train_test_split(x_data, y_data,
                                                                                        test_size=0.2)  # , shuffle=False)

            y_test = [tuple(x) for x in y_test.to_numpy()]
            dt = np.dtype('bool,float')
            y_test = np.asarray(y_test, dt)

            # normalize
            """std = x_train_data.std()
            x_valid_data = (x_valid_data - x_train_data.mean()) / x_train_data.std()
            x_test_data = (x_test_data - x_train_data.mean()) / x_train_data.std()
            x_train_data = (x_train_data - x_train_data.mean()) / x_train_data.std()"""

            # exit()

            # random forest
            # print('csa ' + label + str(int(year / 12)) + 'y')
            # l1_list = [0.01, 0.1, 0.5, 0.8]
            l1_list = [0.8]

            max_auc = 0

            dict_valid_line = {}
            for l1 in l1_list:
                auc_valid_list = []
                correlation_valid_list = []

                count_y = 0
                count_n = 0
                algo_name = 'cox- ' + label
                params = 'depth-' + str(l1)
                # print(params)

                for k in range(10):
                    # for k in range(2):

                    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train_all, y_train_all,
                                                                                          test_size=0.2)

                    # print("No val")
                    # X_train = X_valid = X_train_all
                    # y_train = y_valid = y_train_all

                    X_train_my = pd.concat([X_train, y_train], axis=1)
                    X_valid_my = pd.concat([X_valid, y_valid], axis=1)

                    y_train = [tuple(x) for x in y_train.to_numpy()]
                    dt = np.dtype('bool,float')
                    y_train = np.asarray(y_train, dt)

                    y_valid = [tuple(x) for x in y_valid.to_numpy()]
                    dt = np.dtype('bool,float')
                    y_valid = np.asarray(y_valid, dt)

                    rsf = CoxPHSurvivalAnalysis(alpha=l1).fit(X_train, y_train)

                    # cph = CoxPHFitter()
                    # cph.fit(X_train_my, duration_col='Day' + task_name, event_col="Flag" + task_name)
                    # cph.print_summary()
                    ##train auc
                    surv = rsf.predict_cumulative_hazard_function(X_train)
                    roc_auc_train = calc_auc(surv, y_train, year)

                    ##valid auc
                    surv = rsf.predict_cumulative_hazard_function(X_valid)
                    roc_auc_valid = calc_auc(surv, y_valid, year)
                    correlation_valid = calc_correlation(surv, y_valid, year)
                    auc_valid_list.append(roc_auc_valid)
                    correlation_valid_list.append(correlation_valid)
                    day_prediction = [y[1] for y in y_valid]

                    # roc_auc_valid = sum(auc_valid_list) / 5
                    # roc_auc_valid = auc_valid_list
                results_df.loc[task_name + "_mean"][types_to_take] = np.mean(auc_valid_list)
                results_df.loc[task_name + "_std"][types_to_take] = np.std(auc_valid_list)
                ##test auc
                y_train = [tuple(x) for x in y_train_all.to_numpy()]
                dt = np.dtype('bool,float')
                y_train = np.asarray(y_train, dt)
                rsf = CoxPHSurvivalAnalysis(alpha=l1).fit(X_train_all, y_train)
                # save model
                pickle.dump(rsf, open("models/" + algo_name + ".pkl", 'wb'))
                ##train auc
                surv = rsf.predict_cumulative_hazard_function(X_train_all)
                roc_auc_train = calc_auc(surv, y_train, year)

                surv = rsf.predict_cumulative_hazard_function(X_test)
                roc_auc = calc_auc(surv, y_test, year)

                # print(f'{label} & {round(roc_auc_train,3)} & {round(roc_auc_valid,3)} \pm {round(statistics.pstdev(auc_valid_list),3)} & {round(roc_auc,3)} \\\\ {params}')
                # dict_valid_line[round(roc_auc_valid,3)] = f'{label} & {round(roc_auc_train,3)} & {round(roc_auc_valid,3)} \pm {round(statistics.pstdev(auc_valid_list),3)} & {round(roc_auc,3)} \\\\'
                # dict_valid_line[f'{label} & {round(roc_auc_train,3)} & $ {round(roc_auc_valid,3)} \pm {round(statistics.pstdev(auc_valid_list),3)} $ & {round(roc_auc,3)} \\\\'] = round(roc_auc_valid,3)
                # dict_valid_line[str(round(roc_auc, 3))] = round(roc_auc_valid, 3)

                # print(str(count_y) + ' ' + str(count_n)+ " score:" +  str(rsf.score(X_test, y_test)) )

                max_auc = max(roc_auc, max_auc)

                # plt.step(rsf.event_times_, s, where="post", label=str(i))

            # max_auc = max(auc, max_auc)

            # print(max(dict_valid_line.items(), key=operator.itemgetter(1))[0])
            # list_test[label].append(max(dict_valid_line.items(), key=operator.itemgetter(1))[0])
            columns = X_train_all.columns
            c_index = rsf.score(X_train_all, y_train)
            coeffs = pd.Series({X_train_all.columns[i]: rsf.coef_[i] for i in range(len(X_train_all.columns)) if
                                X_train_all.columns[i] in coeffs_df.index})

            # add this column to the dataframe
            coeffs_df[task_name] = coeffs
            auc_val_cox[task_name] += auc_valid_list
            corr_val_cox[task_name] += correlation_valid_list

            # print('clf: ' + label + ' ' + str(max_auc))
            # print('### ' + label + ' random_forest max_auc: ' +  str(max_auc) + max_params)
            # file_res.close()
        # print('--\t--\t--\t--\t--\t--\t--\t--\t--\t--\t\n')
        cox_dict_corr[ind] = corr_val_cox
        cox_dict[ind] = auc_val_cox

        # corr_df = pd.DataFrame.from_dict(corr_val_cox)
        # corr_df.to_csv(f'cox_demographic_corr.csv')
        # auc_df = pd.DataFrame.from_dict(auc_val_cox)
        # auc_df.to_csv(f'cox_demographic_auc.csv')
        # results_df.to_csv(f'cox_auc_l1_{l1_list[0]}.csv')

    # save coeffs
    coeffs_df.to_csv('cox_coeff_HLA_2205.csv', float_format="%.2f")
    # main_script.createCrossValidResults(cox_dict, "cox")
    # plot_heatmap(coeffs_df, "cox_coeff_diff_without_sparse3.png")
    return coeffs_df

if __name__ == '__main__':
    feature_pvalue = pd.read_csv("pvalue_cox2.csv", index_col=0)
    # write to csv 2 digits after the decimal point
    # coeff = calc_coeff()
    pdict = calc_cox_with_pvalue()
    # coeffs_df = pd.read_csv('cox_coeff_HLA_to_diff_without_sparse2.csv', index_col=0)
    # pdict = pd.read_csv('pvalue_HLA_to_diff_without_sparse2.csv', index_col=0)
    pdict.to_csv('pvalue_HLA_to_diff_without_sparse2.csv', float_format="%.2f")
    # zero all pvalues that are not significant
    # pdict = pdict.fillna(1)
    # for feature in coeffs_df.index:
    #     coeffs_df.loc[feature][pdict.loc[feature] > 0.05] = 0
    # plot_heatmap(coeffs_df, "cox_coeff_diff_without_sparse2.png")
    # exit(1)





    # set results
# print((' & ').join(list_test) + ' \\\\')

# print('-------------------------------------------------------------------------------------\n')




