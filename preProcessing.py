from data_functions import get_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

path_to_dir = "/home/kanna/PycharmProjects/Gen2Vec/"
def replace_HLA_all_data(data_heder, data):
    cols_heder = data_heder.columns
    specipic_names = data.columns
    unique_HLA_cols = set([col.split("-")[0] for col, heder in zip(specipic_names, cols_heder) if "HLA" in heder])
    # replace all HLA
    for col in unique_HLA_cols:
        data[col + "-3"] = data[col + "-3"] - data[col + "-2"]
        data[col + "-2"] = data[col + "-2"] - data[col + "-1"]
    return data

def replace_HLA_akiva_data(data):
    cols = data.columns
    # HLA cols are the cols that divided to 3
    divided_cols = [col.split("-")[0] for col in cols]
    unique_HLA_cols = set([col for col in divided_cols if divided_cols.count(col) == 3])
    # replace all HLA
    for col in unique_HLA_cols:
        data[col + "-3"] = data[col + "-3"] - data[col + "-2"]
        data[col + "-2"] = data[col + "-2"] - data[col + "-1"]
    return data

def matchdataframes(data1, data2):


    # match dataframes based on their results
    common_cols = list(data1.columns.intersection(data2.columns))

    # index_list = df[(df['A'] == 2) & (df['B'] == 3)].index.tolist()

    all_df = pd.merge(data1, data2, on=common_cols, how='left', indicator='exists')
    all_df['exists'] = np.where(all_df.exists == 'both', True, False)
    # return dataframe where both dataframes have the same values in common columns
    return all_df[all_df.exists == True]


def get_sum_hla(dataframe):
    columns_to_num = {col: col.split("-")[1] for col in dataframe.columns}
    new_df = pd.DataFrame()
    for num in columns_to_num.values():
        cols = [col for col in columns_to_num if columns_to_num[col] == num]
        new_df["HLA-" + num] = dataframe[cols].sum(axis=1)
    return new_df


def get_sum_kir(dataframe):
    cols_types = ["L.PT", "L.DNR", "S.PT", "S.DNR"]
    columns_to_LS = {col: col.split(".")[0][2] for col in dataframe.columns}
    columns_to_PD = {col: col.split(".")[1] for col in dataframe.columns}

    columns_to_num = {col: col.split(".")[0][-1] for col in dataframe.columns}
    unique_nums = set(columns_to_num.values())
    final_columns = ["L.PT." + num for num in unique_nums] + ["L.DNR." + num for num in unique_nums] + \
                      ["S.PT." + num for num in unique_nums] + ["S.DNR." + num for num in unique_nums]
    new_df = pd.DataFrame(columns=final_columns)
    for col in final_columns:
        [LS, PD, num] = col.split(".")
        # sum all raws with those values
        cols = [col for col in columns_to_num if columns_to_LS[col] == LS and columns_to_PD[col] == PD and columns_to_num[col] == num]
        new_df[col] = dataframe[cols].sum(axis=1)
    return new_df


def get_labels(labels_dataframe):
    # labels_columns = ['DelSur', 'FlagSur', 'DelGVH2', 'FlagGVH2', 'DelGVH3', 'FlagGVH3',
    #                   'DelCGVH', 'FlagCGVH', 'DelDFS', 'FlagDFS']

    # labels_dataframe = dataframe[[col for col in dataframe if col in labels_columns]]
    labels_dataframe = labels_dataframe.replace(".", 5)
    labels_dataframe = labels_dataframe.replace(-1, 5)
    labels_dataframe = labels_dataframe.replace("-1", 5)

    # labels_dataframe = labels_dataframe.rename(
    #     columns={col: "Del" + col[4:] if "Flag" in col else col for col in labels_dataframe}, inplace=False)

    # add flag for events
    for col in labels_dataframe.columns:
        if "Del" in col:
            labels_dataframe[col[3:] + "_flag"] = labels_dataframe[col].apply(lambda x: 0 if x == 2 or x == 5 else 1)
    return labels_dataframe


def get_input_dataframe(df, types_to_take, features_types_dict):
    columns_to_take = [col for col in df if features_types_dict[col] in types_to_take]
    df = df[columns_to_take]
    if "SUM" in types_to_take:
        if "HLA" in types_to_take:
            hla_cols = [col for col in df if features_types_dict[col] == "HLA"]
            hla_df = df[hla_cols]
            # drop hla_cols
            df = df.drop(hla_cols, axis=1)
            # add sum
            hla_sum_df = get_sum_hla(hla_df)
            new_df = pd.concat([df, hla_sum_df], axis=1)
        if "KIR" in types_to_take:
            kir_cols = [col for col in df if features_types_dict[col] == "KIR"]
            kir_df = df[kir_cols]
            # add sum
            kir_sum_df = get_sum_kir(kir_df)
            new_df = pd.concat([new_df, kir_sum_df], axis=1)
        return new_df
    else:
        return df


def feature2type():
    def get_emb_group(symbol):
        try:
            group = str(int(symbol))
        except:
            group = "None"
        return group

    df = pd.read_csv(f"{path_to_dir}290123/Akiva_Data.csv")
    features_names = df.iloc[0]
    feature2typedict = {feature:ftype.split(".")[0] for ftype, feature in features_names.items()}

    df = pd.read_csv(f"{path_to_dir}290123/All_Data.csv", index_col=0, header=0)
    features_names = df.iloc[1]
    feature2embedding = {feature: get_emb_group(ftype[ftype.find("Emb")+3]) for ftype, feature in features_names.items()}

    return feature2typedict, feature2embedding

def remove_sparse_columns(df):
    cols_to_remain = [col for col in df if len(col.split("-")) > 1 and col.split("-")[1] in ["1", "2", "3"]]
    # remove columns that contain the same value except 10% of the rows
    col_to_drop = [col for col in df if df[col].value_counts(normalize=True).values[0] > 0.95 and len(np.unique(df[col])) < 10]
    col_to_drop = [col for col in col_to_drop if col not in cols_to_remain]
    df = df.drop(col_to_drop, axis=1)
    return df

def get_processed_data(types = ["Demographics"]):

    input_train = pd.read_csv(f"{path_to_dir}290123/Input_train_Basic_dx_HLA_KIR.csv")
    input_test = pd.read_csv(f"{path_to_dir}290123/Input_test_Basic_dx_HLA_KIR.csv")

    output_train = pd.read_csv(f"{path_to_dir}290123/Output_train.csv")
    output_test = pd.read_csv(f"{path_to_dir}290123/Output_test.csv")

    features_types_dict, feature2embeddingdict = feature2type()

    # X = get_input_dataframe(input_train, types_to_take=["Demographics", "HLA", "KIR"], features_types_dict=features_types_dict)
    X = get_input_dataframe(input_train, types_to_take=types, features_types_dict=features_types_dict)

    y = get_labels(output_train)

    X = features_pre_process(X)
    # X_test = preProcessing.features_pre_process(X_test)
    X_trainind, X_testind, y_train, y_val = train_test_split(
        X.index, y, test_size=0.3, random_state=42)

    X_val = X.loc[X_testind]
    X_train = X.loc[X_trainind]
    y_val = y.loc[X_testind]
    y_train = y.loc[X_trainind]

    y_train = labels_pre_process(y_train)
    y_val = labels_pre_process(y_val)
    return X_train, y_train, X_val, y_val


def get_processed_train_and_test(types = ["Demographics"]):
    input_train = pd.read_csv(f"{path_to_dir}/290123/Input_train_Basic_dx_HLA_KIR.csv")
    input_test = pd.read_csv(f"{path_to_dir}290123/Input_test_Basic_dx_HLA_KIR.csv")

    output_train = pd.read_csv(f"{path_to_dir}290123/Output_train.csv")
    output_test = pd.read_csv(f"{path_to_dir}290123/Output_test.csv")

    features_types_dict, feature2embeddingdict = feature2type()

    # X = get_input_dataframe(input_train, types_to_take=["Demographics", "HLA", "KIR"], features_types_dict=features_types_dict)
    X_train = get_input_dataframe(input_train, types_to_take=types, features_types_dict=features_types_dict)
    y_train = get_labels(output_train)

    X_test = get_input_dataframe(input_test, types_to_take=types, features_types_dict=features_types_dict)
    y_test = get_labels(output_test)

    X_train = features_pre_process(X_train)
    X_test = features_pre_process(X_test)

    # add the columns that are in train and not in test
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    # ensure the order of column in the test set is in the same order than in train set
    X_test = X_test[X_train.columns]

    y_train = labels_pre_process(y_train)
    y_test = labels_pre_process(y_test)
    return X_train, y_train, X_test, y_test

def get_all_test():
    input_test = pd.read_csv(f"{path_to_dir}290123/Input_test_Basic_dx_HLA_KIR.csv")
    features_types_dict, feature2embeddingdict = feature2type()

    X_test = get_input_dataframe(input_test, types_to_take=["Demographics", "HLA", "KIR"], features_types_dict=features_types_dict)
    return X_test

def get_all_data():
    input_test = pd.read_csv(f"{path_to_dir}290123/Input_test_Basic_dx_HLA_KIR.csv")
    features_types_dict, feature2embeddingdict = feature2type()

    X_test = get_input_dataframe(input_test, types_to_take=["Demographics", "HLA", "KIR"],
                                 features_types_dict=features_types_dict)

    input_train = pd.read_csv(f"{path_to_dir}290123/Input_train_Basic_dx_HLA_KIR.csv")

    X_train = get_input_dataframe(input_train, types_to_take=["Demographics", "HLA", "KIR"],
                                 features_types_dict=features_types_dict)
    return X_train, X_test

# def features_pre_process(features):
#     # Get dummies
#     output_features = features[['Age', 'Sex', 'Donor Age', 'Donor Sex']]
#     input_features = features.drop(['Age', 'Sex', 'Donor Age', 'Donor Sex'], axis=1)
#
#     # replace nan
#     input_features = input_features.apply(replace_invalid_values)
#     output_features[['Age', 'Donor Age']] = output_features[['Age', 'Donor Age']].apply(replace_point_to_average)
#
#     # replace in te average
#     new_features = pd.concat([input_features, output_features], axis=1)
#     features = pd.get_dummies(new_features)
#     # drop columns that unknown
#     cols_to_drop = [col for col in features.columns if "UNKNOWN" in col]
#     features = features.drop(cols_to_drop, axis=1)
#     return features

def features_pre_process(features):
    # Get dummies
    continuous_features = features[['Age', 'Donor Age']]
    input_features = features.drop(['Age', 'Donor Age'], axis=1)

    # replace to string
    input_features = input_features.apply(replace_to_string)
    # replace nan
    continuous_features = continuous_features.apply(replace_point_to_average)
    scaler = StandardScaler()
    continuous_features = pd.DataFrame(scaler.fit_transform(continuous_features), columns=['Age', 'Donor Age'],index=input_features.index)
    input_features = pd.get_dummies(input_features)

    # replace in te average
    new_features = pd.concat([input_features, continuous_features], axis=1)
    return new_features


# def replace_invalid_values(df_column):
#     def replace_to_valid(value):
#         try:
#             newval = str(round(value))
#         except:
#             newval = "UNKNOWN"
#         return newval
#     df_column = df_column.apply(replace_to_valid)
#     return df_column


def replace_to_string(df_column):
    def replace_to_str(value):
        if isinstance(value, str):
            pass
        elif not np.isnan(value):
            value = str(round(value))
        return value
    df_column = df_column.apply(replace_to_str)
    return df_column

def replace_point_to_average(age_df):
    # calculate average
    age_mean = age_df.replace(".", np.nan).dropna().astype(np.float64).mean()
    age_df = age_df.replace(".", age_mean)
    return age_df


def labels_pre_process(labels_df):
    labels_output = []
    labels_options = ["Sur", "GVH2", "GVH3", "CGVH", "DFS"]
    for label in labels_options:
        try:
            labels_output.append(labels_df[["Del" + label, "Day" + label, label + "_flag"]])
        except:
            labels_output.append(labels_df[["Del" + label, "Day" + label, "Flag" +label]])
    return labels_output


def entropy1(labels, base=None):
  value ,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)


if __name__ == '__main__':
    data_features, labels = get_data()
    data_features = features_pre_process(data_features)
    labels = labels_pre_process(labels)

