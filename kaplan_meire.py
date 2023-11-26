import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import preProcessing
import warnings
from sklearn import preprocessing
import pickle
import Models
warnings.filterwarnings("ignore")
categories = ['GVH2', 'GVH3', 'CGVH', 'DFS', 'Sur']

def get_score_cross_validation():
    X_train_all, X_test_all = preProcessing.get_all_data()
    X_all = pd.concat((X_train_all, X_test_all))
    k_cross = 5
    # check if the data is already saved
    X_train, y_train, X_test, y_test = preProcessing.get_processed_train_and_test()
    X = pd.concat((X_train, X_test))
    y = [pd.concat((y_tr, y_te)) for y_tr, y_te in zip(y_train, y_test)]
    ind = np.random.permutation(y[0].shape[0])
    # split ind to k_cross groups
    ind = np.array_split(ind, k_cross)
    # split to groups by k_cross
    X = [X.iloc[i, :] for i in ind]
    y = [[y_task.iloc[i, :] for y_task in y] for i in ind]
    X_test_score = []
    for k in range(k_cross):
        X_test = X[k]
        y_test = y[k]
        X_train = pd.concat([X[i] for i in range(k_cross) if i != k], axis=0)
        y_train = [pd.concat([y[i][task_ind] for i in range(k_cross) if i != k], axis=0) for task_ind in range(len(y[0]))]
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
        for y_tag in y_test:
            X_test_copy = pd.concat([X_test_copy, y_tag], axis=1)
        # union X test with all the inpormation
        X_test_all = X_all.iloc[ind[k], :]
        # concat X_test with X_test_copy and drop common columns
        unique_cols = list(set(X_test_all.columns) - set(X_test_copy.columns))
        X_test_all = X_test_all[unique_cols]
        score_cols = [label + "_score" for label in models.keys()]
        X_test_copy_all = pd.concat([X_test_all, X_test_copy[score_cols]], axis=1)
        for y_tag in y_test:
            X_test_copy_all = pd.concat([X_test_copy_all, y_tag], axis=1)

        # save the data
        X_test_score.append(X_test_copy_all)
        X_test_score1 = pd.concat(X_test_score)
        with open("X_score_all_data.pkl", "wb") as f:
            pickle.dump(X_test_score1, f)
    return
def get_score():
    # check if the data is already saved
    try:
        X_test_copy = pd.read_pickle("X_test_score_demographic_without_processing.pkl")
    except:
        X_train, y_train, X_test, y_test = preProcessing.get_processed_train_and_test()
        # find the columns in X_train that are not in X_test

        auc_val_vec2, auc_train_vec2, correlation_vec, models = Models.base_model.train_model(X_train.values, y_train, X_test.values,
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
        X_test = preProcessing.get_all_test()
        # concat X_test with X_test_copy and drop common columns
        unique_cols = list(set(X_test.columns) - set(X_test_copy.columns))
        X_test = X_test[unique_cols]
        score_cols = [label + "_score" for label in models.keys()]
        X_test_copy_all = pd.concat([X_test, X_test_copy[score_cols]], axis=1)
        for y in y_test:
            X_test_copy_all = pd.concat([X_test_copy_all, y], axis=1)

        # save the data
        with open("X_test_score.pkl", "wb") as f:
            pickle.dump(X_test_copy_all, f)
    return X_test_copy

def split_by_score(X, cat):
    X["score"] = pd.qcut(X[f"{cat}_score"], 2, labels=False)
    return X

def plot_kaplan_by_score():
    X = get_score()
    results = []
    fig, axs = plt.subplots(nrows=len(categories), ncols=1, figsize=(10, 20))
    for i, cat in enumerate(categories):
        X_groups = split_by_score(X, cat)
        results_temp = []
        new_data = X_groups[~X_groups[f'Day{cat}'].isna()]
        T = new_data[f'Day{cat}']
        E = new_data[f'{cat}_flag']
        # fig, ax = plt.subplots()

        for num in np.unique(X_groups["score"]):
            condition = (new_data["score"] == num)
            TP = T[condition]
            EP = E[condition]

            if T.size != 0:
                kmf = KaplanMeierFitter()
                kmf.fit(TP, event_observed=EP, label=num)
                # kmf.plot_survival_function()
                kmf.plot_survival_function(ax=axs[i], ci_show=True, loc=slice(0., 1000.))

                naf = NelsonAalenFitter()
                naf.fit(TP, event_observed=EP)
                results_temp.append(naf.predict(1000))
            else:
                results_temp.append(0)
        results.append(results_temp)

    save = np.asmatrix(results)
    np.savetxt(f"hazard.csv", save, delimiter=",")
    return save
def plot_kaplan_by_cy():
    file = f'data/Hazard_HLA_sum.csv'
    cy = pd.read_csv("290123/All_Data.csv", header=2)["CY"]
    na_values_symbols = [-1]
    data = pd.read_csv(file, delimiter=',', na_values=na_values_symbols)
    data["CY"] = cy


    results = []

    for cat in categories:
        results_cat = []

        new_data = data[~data[f'Day{cat}'].isna()]
        T = new_data[f'Day{cat}']
        E = new_data[f'Flag{cat}']
        fig, ax = plt.subplots()
        for cy_value in np.unique(cy):
            condition = (data["CY"] == cy_value)
            print(f'{cat}_{cy_value}')

            TP = T[condition]
            EP = E[condition]

            if T.size != 0:
                kmf = KaplanMeierFitter()
                kmf.fit(TP, event_observed=EP, label=cy_value)
                kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0., 1000.))

                naf = NelsonAalenFitter()
                naf.fit(TP, event_observed=EP)
                results_cat.append(naf.predict(1000))
            else:
                results_cat.append(0)
        results.append(results_cat)
        plt.title(f'{cat}_cy')
        ax.set(xlabel='Time', ylabel='Survival probability')
        # fig.savefig(f'plots/km_{cat}_cy_{cy_value}.png')
        plt.show()

    save = np.asmatrix(results)
    np.savetxt(f"haz_cy_{cy_value}.csv", save, delimiter=",")

def plot_kaplan_by_score_and_mismatch():
    fig, axs = plt.subplots(nrows=len(categories), ncols=1, figsize=(10, 20))
    for i, cat in enumerate(categories):
        X = get_score()
        # count mismatches
        cols_to_sum = [col for col in X.columns if len(col.split("-")) == 2 and col.split("-")[1] == '1']
        X["mismatch"] = X[cols_to_sum].sum(axis=1)
        X_groups = split_by_score(X, cat)
        new_data = X_groups[~X_groups[f'Day{cat}'].isna()]
        T = new_data[f'Day{cat}']
        E = new_data[f'{cat}_flag']
        # fig, ax = plt.subplots()

        for num in np.unique(X_groups["score"]):
            condition = (new_data["score"] == num)
            TP = T[condition]
            EP = E[condition]

            mis_num = 2 * len(cols_to_sum)
            for mis_condition_flag in ["Full Match", "Partial Match"]:
                if mis_condition_flag == "Full Match":
                    mis_condition = (new_data["mismatch"] == mis_num)
                else:
                    mis_condition = (new_data["mismatch"] != mis_num)
                TP_mis = TP[mis_condition]
                EP_mis = EP[mis_condition]

                if T.size != 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(TP_mis, event_observed=EP_mis, label=f'{num}:{mis_condition_flag}')
                    kmf.plot_survival_function(ax=axs[i], ci_show=True, loc=slice(0., 1000.))

                    naf = NelsonAalenFitter()
                    naf.fit(TP_mis, event_observed=EP_mis)
                else:
                    a=1
        axs[i].set_title(f'{cat}')
        # set y axis to be between 0 and 1 with 0.2 steps
        axs[i].set_yticks(np.arange(0, 1.2, 0.2))
        # delete x label
        axs[i].set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'plots/km_mismatch.png')
    plt.show()

def plot_kaplen_by_most_important_HLA(model="cox"):
    coeff_table = pd.read_csv(f"results/{model}_coeff.csv", index_col=0)
    fig, axs = plt.subplots(nrows=len(categories), ncols=1, figsize=(10, 20))
    for i, cat in enumerate(categories):
        # find the most important HLA
        HLA_list = [index for index in coeff_table.index if len(index.split("-")) == 2 and index !="H-TBI"]
        HLA_value = abs(coeff_table.loc[HLA_list][cat + "_" + model])
        most_important_hla = HLA_value.idxmax()

        X = get_score()
        # union X with its HLA features
        X_groups = split_by_score(X, cat)
        new_data = X_groups[~X_groups[f'Day{cat}'].isna()]
        T = new_data[f'Day{cat}']
        E = new_data[f'{cat}_flag']

        for num in np.unique(X_groups["score"]):
            condition = (new_data["score"] == num)
            TP = T[condition]
            EP = E[condition]
            occurrences = new_data[most_important_hla].value_counts(normalize=True)
            # sort the HLA values and comute the cumulative sum
            occurrences = occurrences.sort_index()
            occurrences = occurrences.cumsum()
            # find the first HLA value that has a cumulative sum of 0.1
            mis_num = occurrences.index[occurrences > 0.1].tolist()[0]

            for hla_condition_flag in [True, False]:
                if hla_condition_flag:
                    mis_condition = (new_data[most_important_hla] <= mis_num)
                else:
                    mis_condition = (new_data[most_important_hla] > mis_num)
                TP_mis = TP[mis_condition]
                EP_mis = EP[mis_condition]

                if T.size != 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(TP_mis, event_observed=EP_mis, label=f'{num}:HLA-{hla_condition_flag}')
                    kmf.plot_survival_function(ax=axs[i], ci_show=True, loc=slice(0., 1000.))

                    naf = NelsonAalenFitter()
                    naf.fit(TP_mis, event_observed=EP_mis)
                else:
                    a = 1
        axs[i].set_title(f'{cat} - {most_important_hla}')
        # set y axis to be between 0 and 1 with 0.2 steps
        axs[i].set_yticks(np.arange(0, 1.2, 0.2))
        # delete x label
        axs[i].set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'plots/km_hla_{model}.png')
    plt.show()


def plot_kaplen_by_most_important_KIR(model="cox"):
    coeff_table = pd.read_csv(f"results/{model}_coeff.csv", index_col=0)
    fig, axs = plt.subplots(nrows=len(categories), ncols=1, figsize=(10, 20))
    for i, cat in enumerate(categories):
        # find the most important HLA
        KIR_list = [index for index in coeff_table.index if len(index.split(".")) == 2]
        HLA_value = abs(coeff_table.loc[KIR_list][cat + "_" + model])
        most_important_kir = HLA_value.idxmax()

        X = get_score()
        # union X with its HLA features
        X_groups = split_by_score(X, cat)
        new_data = X_groups[~X_groups[f'Day{cat}'].isna()]
        T = new_data[f'Day{cat}']
        E = new_data[f'{cat}_flag']

        for num in np.unique(X_groups["score"]):
            condition = (new_data["score"] == num)
            TP = T[condition]
            EP = E[condition]
            occurrences = new_data[most_important_kir].value_counts(normalize=True)
            # sort the HLA values and comute the cumulative sum
            occurrences = occurrences.sort_index()
            occurrences = occurrences.cumsum()
            # find the first HLA value that has a cumulative sum of 0.1
            mis_num = occurrences.index[occurrences > 0.1].tolist()[0]

            for kir_condition_flag in [True, False]:
                if kir_condition_flag:
                    mis_condition = (new_data[most_important_kir] <= mis_num)
                else:
                    mis_condition = (new_data[most_important_kir] > mis_num)
                TP_mis = TP[mis_condition]
                EP_mis = EP[mis_condition]

                if T.size != 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(TP_mis, event_observed=EP_mis, label=f'{num}:KIR-{kir_condition_flag}')
                    kmf.plot_survival_function(ax=axs[i], ci_show=True, loc=slice(0., 1000.))

                    naf = NelsonAalenFitter()
                    naf.fit(TP_mis, event_observed=EP_mis)
                else:
                    a = 1
        axs[i].set_title(f'{cat} - {most_important_kir}')
        # set y axis to be between 0 and 1 with 0.2 steps
        axs[i].set_yticks(np.arange(0, 1.2, 0.2))
        # delete x label
        axs[i].set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'plots/km_kir_{model}.png')
    plt.show()

def plot_permissive():
    y = pd.read_csv(f'Data/All_Data.csv', header=2)
    file = f'Input_Age_HLA_KIR_permis.csv'
    na_values_symbols = [-1]
    data = pd.read_csv(file, delimiter=',', na_values=na_values_symbols)

    data = preProcessing.matchdataframes(data, y)

    for cat in categories:
        results_cat = []

        new_data = data[~data[f'Day{cat}'].isna()]
        T = new_data[f'Day{cat}']
        E = new_data[f'Flag{cat}']
        fig, ax = plt.subplots()
        for value in np.unique(data["Permiss1"]):
            condition = (data["Permiss1"] == value)

            TP = T[condition]
            EP = E[condition]

            if T.size != 0:
                kmf = KaplanMeierFitter()
                kmf.fit(TP, event_observed=EP, label=value)
                kmf.plot_survival_function(ax=ax, ci_show=False, loc=slice(0., 1000.))

                naf = NelsonAalenFitter()
                naf.fit(TP, event_observed=EP)
                results_cat.append(naf.predict(1000))
            else:
                results_cat.append(0)
        plt.title(f'{cat}_permissive')
        ax.set(xlabel='Time', ylabel='Survival probability')
        plt.savefig(f'plots/km_{cat}_permissive.png')
        # fig.savefig(f'plots/km_{cat}_cy_{cy_value}.png')
        plt.show()
        a=1


def plot_kaplan_by_score_and_feature(feature):
    fig, axs = plt.subplots(nrows=len(categories), ncols=1, figsize=(10, 20))
    for i, cat in enumerate(categories):
        X = get_score()
        # count mismatches
        cols_to_sum = [col for col in X.columns if len(col.split("-")) == 2 and col.split("-")[1] == '1']
        X["mismatch"] = X[cols_to_sum].sum(axis=1)
        X = preProcessing.replace_HLA_akiva_data(X)

        X_groups = split_by_score(X, cat)
        new_data = X_groups[~X_groups[f'Day{cat}'].isna()]

        # fig, ax = plt.subplots()
        if new_data[feature].unique().size > 2:

            col_median = new_data[feature].median()
            # replace the values with 0 or 1
            group1 = new_data[new_data[feature] > col_median]
            if group1.shape[0] < 10:
                new_data[feature] = [1 if float(x) >= col_median else 0 for x in new_data[feature]]
            else:
                new_data[feature] = [1 if float(x) > col_median else 0 for x in new_data[feature]]

        T = new_data[f'Day{cat}']
        E = new_data[f'{cat}_flag']
        for num in np.unique(X_groups["score"]):
            condition = (new_data["score"] == num)
            TP = T[condition]
            EP = E[condition]

            for condition_flag in [1, 0]:
                mis_condition = (new_data[feature] == condition_flag)
                TP_mis = TP[mis_condition]
                EP_mis = EP[mis_condition]

                if TP_mis.size != 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(TP_mis, event_observed=EP_mis, label=f'Score:{num}, Match:{condition_flag}')
                    kmf.plot_survival_function(ax=axs[i], ci_show=True, loc=slice(0., 1000.))

                    naf = NelsonAalenFitter()
                    naf.fit(TP_mis, event_observed=EP_mis)
                else:
                    a=1
        axs[i].set_title(f'{cat}-{feature}')
        # set y axis to be between 0 and 1 with 0.2 steps
        axs[i].set_yticks(np.arange(0, 1.2, 0.2))
        # delete x label
        axs[i].set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'plots/km_mismatch_{feature}.png')
    plt.show()


if __name__ == '__main__':
    get_score_cross_validation()
    plot_permissive()
    get_score()
    plot_kaplen_by_most_important_HLA(model="shap")
    plot_kaplen_by_most_important_KIR(model="hazard")

    plot_kaplan_by_score_and_mismatch()
    plot_kaplan_by_score_and_feature("C-1")
    plot_kaplan_by_score_and_feature("C-3")
    plot_kaplan_by_score_and_feature("DRB1-1")
    plot_kaplan_by_score_and_feature("DRB1-3")

    plot_kaplan_by_cy()
    plot_kaplan_by_score()

