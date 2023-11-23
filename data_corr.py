import pandas as pd
import numpy as np
import preProcessing
import seaborn as sns
import matplotlib.pyplot as plt
labels = ["Sur", "GVH2", "GVH3", "CGVH", "DFS"]
data = pd.read_csv("All_Data.csv", index_col=0,header=2)
features_types_dict, feature2embeddingdict = preProcessing.feature2type()
#  take the 3-rd raw and beyond
data = data.iloc[2:]

# remove columns with the same value
data = data.loc[:, (data != data.iloc[0]).any()]

#     remove output columns
output_columns = [col for col in data.columns if np.any([label in col for label in labels])]
y = data[["Del" + label for label in labels]]
data.drop(output_columns, axis=1, inplace=True)

# remove HLA columns
HLA_columns = [col for col in data.columns if "HLA" == features_types_dict[col]]
HLA_df = data[HLA_columns]
data.drop(HLA_columns, axis=1, inplace=True)

corr_df = pd.DataFrame(index=data.columns, columns=[labels + HLA_columns])
# find the correlation between the features to y columns
for col in data.columns:
    for label in labels:
        corr_df.loc[col, label] = data[col].corr(y["Del" + label])
    for hla_col in HLA_columns:
        corr_df.loc[col, hla_col] = data[col].corr(HLA_df[hla_col])

# change the index to the feature name to be without brackets
corr_df.index = [col.split("(")[0] for col in corr_df.index]
# replace nan with 0
corr_df.fillna(0, inplace=True)
# plot heatmap of the correlation
sns.heatmap(corr_df, cmap="coolwarm",
            xticklabels=True, yticklabels=True)
# plt.savefig("plots/correlation.png")
plt.tight_layout()
# rotate x labels
plt.xticks(rotation=90)

# decrease the size of the labels
plt.tick_params(axis='both', which='major', labelsize=5)
plt.show()
a=1