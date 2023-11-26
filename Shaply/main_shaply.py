import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from model import NeuralNetwork
import preProcessing
import shap
import warnings
from TrainTest import test
warnings.filterwarnings("ignore")
import seaborn as sns

best_params = {"dropout_prob": 0.0,
               'initialization': True,
               "alpha": 0.5,
               "hidden1": 10,
               "hidden2": 8}

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_output():
    for category in ['Sur', 'GVH2', 'GVH3', 'CGVH', 'DFS']:
        # category = 'Sur'
    # category = 'Sur'
        suffix = 'Basic_dx_HLA_KIR'
        param_set = 'KIR'

        file_input_train = f"./Data/Input_train_{suffix}.csv"
        file_input = f"Input_train_{suffix}.csv"
        file_label = f"./Data/Output_train.csv"
        binary_predict = True

        params = {'output_size': 1,
                  'num_output': 1,
                  'cuda_device': 0}

        params.update(best_params)

        # Get cpu or gpu device for training.
        set_seed(0)

        # load data
        pre_data_x_train = pd.read_csv(file_input_train, delimiter=',')
        pre_data_x = pd.read_csv(file_input, delimiter=',')
        pre_data_x_train = preProcessing.replace_HLA(pre_data_x_train)
        pre_data_x = preProcessing.replace_HLA(pre_data_x)


        data_x = pre_data_x.values
        input_size = data_x.shape[1]

        pre_data_y = pd.read_csv(file_label, delimiter=',')
        data_y_bin = pre_data_y[f'Del{category}'].values
        data_y_cont = pre_data_y[f'Day{category}'].values
        data_flag = pre_data_y[f'Flag{category}'].values

        data_y_cont_log = [np.log(data_y_cont[i]) if data_flag[i] == 1 else 0 for i in range(len(data_flag))]
        data_var = [data_y_cont_log[i] for i in range(len(data_flag)) if data_flag[i] == 1]
        var = np.var(data_var)

        data_x = torch.tensor(data_x, dtype=torch.float)
        data_y_bin = torch.tensor(data_y_bin, dtype=torch.float)
        data_y_cont_log = torch.tensor(data_y_cont_log, dtype=torch.float)
        data_flag = torch.tensor(data_flag, dtype=torch.float)

        test_dataset = TensorDataset(data_x, data_y_bin, data_y_cont_log, data_flag)
        test_dataloader = DataLoader(test_dataset)

        # Compute output
        dims = [input_size, params['hidden1'], params['hidden2'], params['output_size']]
        model = NeuralNetwork(dims, params['num_output'], params['dropout_prob'], params['initialization']).to(device)
        loss, auc, corr = test(test_dataloader, model, best_params['alpha'], var, device)

        # Print output
        print(f'Auc is {auc:.04f} | Corr is {corr:.04f} | Loss is {loss:.04f}')

        # Compute Shapley graph
        explainer = shap.Explainer(model.predict, pre_data_x_train.values)
        shap_values = explainer(pre_data_x.values)
        shap.summary_plot(shap_values, pre_data_x.values, feature_names=pre_data_x.columns, show=False, plot_size=(12, 8))
        bin_or_cont = 'bin' if binary_predict else 'cont'
        plt.title(f'shap_train_{category}_{suffix}_{bin_or_cont}')
        plt.savefig(f'shap_train_{category}_{suffix}_{bin_or_cont}.png')

        # create csv file of the coefficients
        shap_values_df = pd.DataFrame(shap_values.values, columns=pre_data_x.columns)
        shap_values_df.to_csv(f'shap_values_{category}_{suffix}_{bin_or_cont}.csv')


def process_shap(input_df):
    # normalize the input data
    input_df = (input_df - input_df.mean()) / input_df.std()
    shap_df_all = pd.DataFrame(index=input_df.columns, columns=['Sur_shap', 'GVH2_shap', 'GVH3_shap', 'CGVH_shap', 'DFS_shap'])
    for category in ['Sur', 'GVH2', 'GVH3', 'CGVH', 'DFS']:
        shap_df = pd.read_csv(f'shap_values_{category}_Basic_dx_HLA_KIR_bin.csv', index_col=0)
        mul_df = input_df * shap_df.values
        shap_df_all[category + "_shap"] = mul_df.sum(axis=0)

    shap_df_all.to_csv('shap_df_all.csv')
    return shap_df_all


def print_shap():
    suffix = 'Basic_dx_HLA_KIR'
    type = 'shap'
    scaling = 'rescaled'
    data = pd.read_csv(f'{type}_{suffix}.csv', delimiter=',')
    data = data.iloc[:, :50]
    sns.set(font_scale=1.5)
    # palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    rows = ['GVH2', 'GVH3', 'CGVH', 'Rel', 'Sur']
    plt.rcParams["figure.figsize"] = (12, 12)

    data = data.T
    data.columns = rows
    data = data[['Sur', 'Rel', 'CGVH', 'GVH3', 'GVH2']]
    hmap = sns.heatmap(data=data, vmin=-1.0, vmax=1.0, cmap=plt.get_cmap('bwr'),
                       # xticklabels=rows,
                       # yticklabels=cols,
                       # cbar=False
                       )
    hmap.xaxis.tick_top()
    plt.savefig(f'heatmap_{type}_{suffix}_{scaling}_rel_Basic.png')
    plt.show()




if __name__ == "__main__":
    compute_output()
    suffix = 'Basic_dx_HLA_KIR'
    file_input = f"Input_train_{suffix}.csv"
    bin_or_cont = 'bin'
    input_df = pd.read_csv(file_input, delimiter=',')
    data = process_shap(input_df)
    print_shap(data)
