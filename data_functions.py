import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import itertools
input_file_name = "20211222_GWAS-HCT-result-v2.xlsx"
output_file_name = "gwas_endponts for dan.xlsx"
akiva_input_file = "Input_Output_Age_HLA_KIR_median_Bw.xlsx"
new_file_name = "gwas_data_8dec22.xlsx"

embedding_columns = ["FCGR2A-Allele1",
                        "FCGR2A-Allele2",
                        "FCGR2A-Condon-A1",
                        "FCGR2A-Condon-A2",
                        "FCGR3A-G",
                        "FCGR3A-T",
                        "FCGR3A-Condon-A1",
                        "FCGR3A-Condon-A2"
                    ]


def new_input_file_processing(filename=new_file_name):
    """
    gwas_data_8dec22
    :param file_name:
    :return:
    """
    relevant_columns = ["UPN", "dontype", "pntcmv", "doncmv", "prep", "minitx 0=no 1=yes", "agvh prophy", "dxgroup"]
    input_dataframe = pd.read_excel(f'data/{filename}', usecols=relevant_columns, index_col=0)

    # split
    columns_to_split = ["prep", "agvh prophy"]
    multilabelsmatrix = []
    for column_to_split in columns_to_split:
        col = input_dataframe[column_to_split]
    #     find the  different disease
        col_disease = [[re.sub("[\(\[].*?[\)\]]", "", string) for string in x.split(",")] if isinstance(x, str) else [] for x in col]
        diseases_names = np.unique(list(itertools.chain(*col_disease)))
        multilabel = pd.DataFrame(0, index=input_dataframe.index, columns=diseases_names)
    #     build dataframe to each disease and add them
        for disease in diseases_names:
            flag_list = [1 if disease in x else 0 for x in col_disease]
            multilabel[disease] = pd.DataFrame(flag_list, index=input_dataframe.index)
        multilabelsmatrix.append(multilabel)

    #  assign it in the original input
    input_dataframe = input_dataframe.drop(columns_to_split, axis=1)
    input_dataframe = pd.get_dummies(input_dataframe)
    for multilabel in multilabelsmatrix:
        input_dataframe = pd.concat((input_dataframe, multilabel), axis=1)
    return input_dataframe


def get_new_data(filename=new_file_name, outputfilename=output_file_name):
    data_features = new_input_file_processing(filename)
    demographic_features = get_demographic_features()
    common_indexes = data_features.index.intersection(demographic_features.index)
    data_features = pd.concat((data_features.loc[common_indexes], demographic_features.loc[common_indexes]), axis=1)
    labels = read_output(outputfilename).loc[common_indexes]

    return data_features, labels


def get2301data(outputfilename=output_file_name):
    import preProcessing
    import process_8dec22_data
    data_features = process_8dec22_data.get_data()
    demographic_features = get_demographic_features()
    demographic_features = preProcessing.features_pre_process(demographic_features)
    common_indexes = data_features[0].index.intersection(demographic_features.index)
    data_features = [features.loc[common_indexes] for features in data_features]
    demographic_features = demographic_features.loc[common_indexes]
    data_features.append(demographic_features)
    # data_features = pd.concat((data_features.loc[common_indexes], demographic_features.loc[common_indexes]), axis=1)
    labels = read_output(outputfilename).loc[common_indexes]

    return data_features, labels

def get_AKIVA_input():
    input_dataframe = pd.read_excel(f'data/{akiva_input_file}', sheet_name="Input Final", index_col=0)
    return input_dataframe


def get_embedding_features(filename=input_file_name):
    relevant_columns = ['ID-pair', 'PT_DNR', 'Bw4/6', 'Bw4/6.1', 'C1/C2',
     'C1/C2.1', 'FCGR2A-Allele1', 'FCGR2A-Allele2',
     'FCGR2A-Condon-A1', 'FCGR2A-Condon-A2', 'FCGR3A-G', 'FCGR3A-T',
     'FCGR3A-Condon-A1', 'FCGR3A-Condon-A2', 'MICA_codon129-allel1', 'MICA_codon129-allel2',
     'Centromere-motif1',
     'Centromere-motif2', '2DL1',
     '2DL2', '2DL3', '2DL4', '2DL5', '2DS1', '2DS2', '2DS3', '2DS4', '2DS5',
     '3DL1', '3DL2', '3DL3', '3DS1', '2DP1', '3DP1']

    input_dataframe = pd.read_excel(f'data/{filename}', usecols=relevant_columns, index_col=0)
    patients = input_dataframe['PT_DNR'] == "PT"
    input_dataframe = input_dataframe.drop(['PT_DNR'], axis=1)
    patients_dataframe = input_dataframe[patients]
    donors_dataframe = input_dataframe[~patients]

    patients_dataframe.columns = [col + "_patient" for col in patients_dataframe.columns]
    donors_dataframe.columns = [col + "_donor" for col in donors_dataframe.columns]
    features_dataframe = pd.concat([patients_dataframe, donors_dataframe], axis=1)
    features_dataframe.columns = [col + "_emb" for col in features_dataframe.columns]

    return features_dataframe



def get_data(inputfilename=input_file_name, outputfilename=output_file_name, embedding_flag=False):
    """
    Restore Akiva pre processing
    :param inputfilename:
    :param outputfilename:
    :return:
    """
    # Take data from output file
    data_features = input_file_processing(inputfilename, embedding_flag)
    demographic_features = get_demographic_features()
    common_indexes = data_features.index.intersection(demographic_features.index)
    data_features = pd.concat((data_features.loc[common_indexes], demographic_features.loc[common_indexes]), axis=1)
    labels = read_output(outputfilename).loc[common_indexes]
    return data_features, labels


def get_demographic_features(output_file_name=output_file_name):
    # output inputs
    output_features = pd.read_excel(f'data/{output_file_name}',
                                usecols=['UPN', 'Age', 'Sex', 'Donor Age', 'Donor Sex'], index_col=0)
    return output_features


def read_output(file_name=output_file_name):
    data = pd.read_excel(f'data/{file_name}',
                       usecols=['UPN', 'DelSur', 'DaySur', 'DelGVH2', 'DayGVH2', 'DelGVH3', 'DayGVH3',
                       'DelCGVH', 'DayCGVH', 'DelDFS', 'DayDFS'],
                       index_col=0)
    # ignore invalid
    data = data.replace(".", 5)
    # add flag for events
    for col in data.columns:
        if "Del" in col:
            data[col[3:] + "_flag"] = data[col].apply(lambda x: 0 if x == 2 or x == 5 else 1)
    return data


def input_file_processing(filename=input_file_name, embedding_flag=False):
    relevant_columns = ["ID-pair",
                        "PT_DNR",
    "Bw4/6",
    "Bw4/6",
    "C1/C2",
    "C1/C2",
    "A-allele1",
    "A-allele2",
    "B-allele1",
    "B-allele2",
    "C-allele1",
    "C-allele2",
    "DPA1-allele1",
    "DPA1-allele2",
    "DPB1-allele1",
    "DPB1-allele2",
    "DQA1-allele1",
    "DQA1-allele2",
    "DQB1-allele1",
    "DQB1-allele2",
    "DRB1-allele1",
    "DRB1-allele2",
    "DRB345-allele1",
    "DRB345-allele2",
    "2DL1",
    "2DL2",
    "2DL3",
    "2DL4",
    "2DL5",
    "2DS1",
    "2DS2",
    "2DS3",
    "2DS4",
    "2DS5",
    "3DL1",
    "3DS1"]

    if embedding_flag:
        relevant_columns = relevant_columns + embedding_columns

    input_dataframe = pd.read_excel(f'data/{filename}', usecols=relevant_columns, index_col=0)

    # remove single raws
    IDs = list(input_dataframe.index)
    single_id = [i for i in IDs if IDs.count(i)==1]
    input_dataframe = input_dataframe.drop(single_id)

    # split to donor and patients
    patients = input_dataframe['PT_DNR'] == "PT"
    patients_dataframe = input_dataframe[patients]
    donors_dataframe = input_dataframe[~patients]

    # get the alleles
    patients_alleles1, patients_alleles2 = take_and_split_alleles(patients_dataframe)
    donors_alleles1,  donors_alleles2 = take_and_split_alleles(donors_dataframe)

    # check matching
    match_alleles = check_alleles_matching(patients_alleles1, patients_alleles2, donors_alleles1, donors_alleles2)

    # KIR
    cols = [col for col in relevant_columns if "DS" in col or "DL" in col]
    KIR_patients = patients_dataframe[cols]
    KIR_donors = donors_dataframe[cols]
    # change names
    KIR_patients.columns = [col + "_patient" for col in cols]
    KIR_donors.columns = [col + "_donor" for col in cols]

    if embedding_flag:
    # embedding
        emb_patients = patients_dataframe[embedding_columns]
        emb_donors = donors_dataframe[embedding_columns]
        # change names
        emb_patients.columns = [col + "_embedding" for col in embedding_columns]
        emb_donors.columns = [col + "_embedding" for col in embedding_columns]




    relation = donors_dataframe["PT_DNR"].apply(lambda x: 1 if "URD" in x else 0)
    relation = pd.DataFrame(data=relation, index=donors_dataframe.index)

    # common = input_dataframe.index.intersection(output_features.index)

    # union all features
    # features = pd.concat([match_alleles.loc[common], KIR_patients.loc[common],
    #                       KIR_donors.loc[common], relation.loc[common],
    #                       output_features.loc[common]], axis=1)

    features = pd.concat([match_alleles, KIR_patients,
                          KIR_donors, relation], axis=1)

    if embedding_flag:
        features = pd.concat([features, emb_patients,  emb_donors], axis=1)

    return features


def take_and_split_alleles(dataframe):
    new_dataframe = pd.DataFrame()
    for column in dataframe.columns:
        if "allele" in column:
            allele_dataframe = dataframe[column].str.split(':', expand=True)
            field1 = allele_dataframe[0]
            field2 = field1 + ":" + allele_dataframe[1]
            field3 = field2 + ":" + allele_dataframe[2]
            new_dataframe[[column+"_field1", column+"_field2", column+"_field3"]] = pd.DataFrame(list(zip(field1, field2, field3)), index=dataframe.index)
    # split to allele 1 and allele 2
    allele1 = [col for col in new_dataframe.columns if "allele1" in col]
    allele2 = [col for col in new_dataframe.columns if "allele2" in col]
    return new_dataframe[allele1], new_dataframe[allele2]


def check_alleles_matching(patient_allele1, patient_allele2, donors_alleles1, donors_alleles2):
    regular_matching_dataframe = pd.DataFrame(index=patient_allele1.index)
    cross_matching_dataframe = pd.DataFrame(index=patient_allele1.index)
    matching_dataframe = pd.DataFrame(index=patient_allele1.index)
    for column1, column2 in zip(patient_allele1.columns, patient_allele2.columns):
        regular_matching_dataframe[column1 + "1"] = np.where(patient_allele1[column1] == donors_alleles1[column1], 1, 0)
        regular_matching_dataframe[column1 + "2"] = np.where(patient_allele2[column2] == donors_alleles2[column2], 1, 0)

        cross_matching_dataframe[column1 + "1"] = np.where(patient_allele1[column1] == donors_alleles2[column2], 1, 0)
        cross_matching_dataframe[column1 + "2"] = np.where(patient_allele2[column2] == donors_alleles1[column1], 1, 0)

        # Check how many connect - 0, 1 or 2
        regular_matching_dataframe[column1 + "_match"] = regular_matching_dataframe[column1 + "1"] + regular_matching_dataframe[column1 + "2"]
        cross_matching_dataframe[column1 + "_match"] = cross_matching_dataframe[column1 + "1"] + cross_matching_dataframe[column1 + "2"]

        # Take the max for matching
        name = column1.split("-")[0] + "-" + column1[-1]
        matching_dataframe[name] = np.maximum(regular_matching_dataframe[column1 + "_match"], cross_matching_dataframe[column1 + "_match"])
    return matching_dataframe



if __name__ == '__main__':
    # x = read_output_file()
    # a=1
    get_data()