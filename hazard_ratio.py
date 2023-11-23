import pandas as pd
import numpy as np
from lifelines import NelsonAalenFitter
import preProcessing
import warnings
from lifelines.statistics import logrank_test
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats as stats
warnings.filterwarnings("ignore")
path_to_dir = "/home/kanna/PycharmProjects/Gen2Vec/"

def plot_heatmap(haz_df):
    # replace 0 with exp(-1)
    haz_df = haz_df.replace(0, np.exp(-1))
    # do log transform
    haz_df = np.log(haz_df)
    data_heder = pd.read_csv(f"{path_to_dir}All_Data.csv", header=1)

    columns_HLA = [data_heder.loc[0, col] for col in data_heder.columns if
                   'HLA' in col and data_heder.loc[0, col] in haz_df.index]
    df_to_plot = haz_df.loc[columns_HLA, :]
    f = sns.heatmap(df_to_plot, annot=False, cmap="coolwarm", cbar=False, vmin=-1, vmax=1)
    plt.tight_layout()
    f.set_xticklabels([col.split("_")[0] for col in df_to_plot.columns], rotation=0)
    plt.savefig(f"plots/hazard + '_HLA_nosparse.png")
    plt.show()



    columns_KIR = [data_heder.loc[0, col] for col in data_heder.columns if
                   'KIR' in col and data_heder.loc[0, col] in haz_df.index]
    df_to_plot = haz_df.loc[columns_KIR, :]
    f = sns.heatmap(df_to_plot, annot=False, cmap="coolwarm", cbar=False, vmin=-1, vmax=1)
    plt.tight_layout()
    f.set_xticklabels([col.split("_")[0] for col in df_to_plot.columns], rotation=0)

    plt.savefig(f"plots/hazard + '_KIR_nosparse.png")
    plt.show()

    columns_DEOGRAPICS = [data_heder.loc[0, col] for col in data_heder.columns if
                          'Demographics' in col and data_heder.loc[0, col] in haz_df.index]
    df_to_plot = haz_df.loc[columns_DEOGRAPICS, :]
    f = sns.heatmap(df_to_plot, annot=False, cmap="coolwarm", cbar=False, vmin=-1, vmax=1)
    plt.tight_layout()
    f.set_xticklabels([col.split("_")[0] for col in df_to_plot.columns], rotation=0)
    plt.savefig(f"plots/hazard + '_DEMO_nosparse.png")
    plt.show()

def calculate_auc(hazard_df):
    test_data = pd.read_csv(f"{path_to_dir}data/Input_test_Basic_dx_HLA_KIR.csv", delimiter=',')
    test_labels = pd.read_csv(f"{path_to_dir}data/Output_test.csv", delimiter=',')
    test_data = preProcessing.replace_HLA_akiva_data(test_data)
    # take only the columns that are in the hazard_df and in the test_data
    common_columns = list(set(hazard_df.index).intersection(set(test_data.columns)))
    test_data = test_data[common_columns]
    hazard_df = hazard_df.loc[common_columns]
    correlation_dict = {}
    for col in hazard_df.columns:
        task_name = col.split("_")[0]
        hazard_coeff = hazard_df[col]
        input_value = hazard_coeff.values * test_data.values
        prediction = input_value.sum(axis=1)
        correlation_dict[task_name] = stats.spearmanr(prediction, test_labels["Day" + task_name])[0]
    corr_df = pd.DataFrame.from_dict(correlation_dict)
    return correlation_dict


if __name__ == '__main__':
    hazard_df = pd.read_csv('haz_ratio_without_sparse.csv', index_col=0)
    calculate_auc(hazard_df)
    exit(1)

    df = pd.read_csv('haz_ratio_without_sparse.csv', index_col=0)
    plot_heatmap(df)
    exit(1)

    na_values_symbols = [-1]
    data = pd.read_csv("290123/Input_train_Basic_dx_HLA_KIR.csv", delimiter=',', na_values=na_values_symbols)
    data = preProcessing.replace_HLA_akiva_data(data)
    data = preProcessing.remove_sparse_columns(data)
    labels = pd.read_csv("290123/Output_train.csv", delimiter=',', na_values=na_values_symbols)
    data = pd.concat((data, labels), axis=1)

    data_test = pd.read_csv("290123/Input_test_Basic_dx_HLA_KIR.csv", delimiter=',', na_values=na_values_symbols)
    data_test = preProcessing.replace_HLA_akiva_data(data_test)
    data_test = preProcessing.remove_sparse_columns(data_test)
    labels_test = pd.read_csv("290123/Output_test.csv", delimiter=',', na_values=na_values_symbols)
    data_test = pd.concat((data_test, labels_test), axis=1)

    data = pd.concat((data, data_test), axis=0)
    # set index to 1 - number of rows
    data.index = np.arange(1, len(data) + 1)

    categories = ['GVH2', 'GVH3', 'CGVH', 'DFS', 'Sur']
    columns = data.columns.values.tolist() #[11:]
    results_haz_1 = []
    results_haz_2 = []
    results_ratio = []
    p_value = []
    for col in columns:

        results_temp_haz_1 = []
        results_temp_haz_2 = []
        results_temp_ratio = []
        p_value_temp = []
        for cat in categories:

            print(f'{cat}_{col}')
            new_data = data[~data[f'Day{cat}'].isna()]
            T = new_data[f'Day{cat}']
            E = new_data[f'Flag{cat}']

            condition = (data[col] == 1)
            TP = T[condition]
            EP = E[condition]
            TN = T[~condition]
            EN = E[~condition]

            if TP.size != 0 and TN.size != 0:
                naf = NelsonAalenFitter()
                naf.fit(TP, event_observed=EP)
                results_temp_haz_1.append(naf.predict(1000))
                naf.fit(TN, event_observed=EN)
                results_temp_haz_2.append(naf.predict(1000))

                # save the ratio of the two
                results_temp_ratio.append(results_temp_haz_1[-1] / results_temp_haz_2[-1])

                p_value_temp.append(logrank_test(TP, TN, event_observed_A=EP, event_observed_B=EN).p_value)

            else:
                results_temp_haz_1.append(0)
                results_temp_haz_2.append(0)
                results_temp_ratio.append(0)
                p_value_temp.append(0)

        results_haz_1.append(results_temp_haz_1)
        results_haz_2.append(results_temp_haz_2)
        results_ratio.append(results_temp_ratio)
        p_value.append(p_value_temp)

    save = np.asmatrix(results_ratio)
    # to data frame
    df = pd.DataFrame(save, index=columns, columns=[cat + "_hazard" for cat in categories])
    df.to_csv(f'haz_ratio_without_sparse.csv')

    # save the p value
    save = np.asmatrix(p_value)
    df = pd.DataFrame(save, index=columns, columns=[cat + "_p_value" for cat in categories])
    df.to_csv(f'haz_p_value_without_sparse.csv')


    # np.savetxt(f'haz_1.csv', save.T, delimiter=",")
    # save = np.asmatrix(results_haz_2)
    # np.savetxt(f'haz_2.csv', save.T, delimiter=",")