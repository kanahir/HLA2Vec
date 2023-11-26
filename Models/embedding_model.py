import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from sklearn import preprocessing
import preProcessing
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, data_for_embedding, labels):
        self.data = torch.tensor(data.values, device=device).float()
        self.n_labels = len(labels)
        self.labels = [torch.tensor(label, device=device).float() for label in labels]

        self.data_for_embedding = [torch.tensor(data.values, device=device).float() for data in data_for_embedding]
        self.n_emb = len(data_for_embedding)

    # Length of the Dataset
    def __len__(self):
        return self.data.shape[0]

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        return self.data[idx, :], \
            [self.data_for_embedding[i][idx, :] for i in range(self.n_emb)], \
            [self.labels[i][idx, :] for i in range(self.n_labels)]


class Model(nn.Module):
    def __init__(self, entrop, var, n_labels, emb_len, input_shape, hidden1, hidden2, params):
        super().__init__()
        self.n_labels = n_labels
        self.alpha = params["alpha"]
        drop = params["dropout"]
        self.data_var = var
        self.entropy = entrop

        self.drop = nn.Dropout(p=drop)

        after_emb_len = [int(round(embbeding_len/3)) for embbeding_len in emb_len]
        self.embedding = [nn.Linear(in_features=embbeding_len, out_features=after_embedding_len)
                          for embbeding_len, after_embedding_len in zip(emb_len, after_emb_len)]

        self.hidden1 = nn.Linear(
            in_features=input_shape + sum(after_emb_len), out_features=hidden1
        )
        self.hidden2 = nn.Linear(
            in_features=hidden1, out_features=hidden2
        )

        self.last_layer = [nn.Linear(
            in_features=hidden2, out_features=2
        ) for _ in range(n_labels)]

        self.normalization_layer = [(nn.Linear(
            in_features=2, out_features=1), nn.Linear(
            in_features=2, out_features=1)) for _ in range(n_labels)]

        self.ce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, x, x_for_embedding):
        x_embedding = torch.concat([self.embedding[i](x_for_embedding[i]) for i in range(len(x_for_embedding))], dim=1)
        x_input = torch.concat((x, x_embedding), dim=1)
        x = self.hidden1(x_input)
        x = self.drop(torch.relu(x))
        x = self.hidden2(x)
        x = self.drop(torch.relu(x))
        # last layer
        pred = [layer(x) for layer in self.last_layer]

        pred = [torch.squeeze(torch.stack([layer[0](mission_pred), layer[1](mission_pred)])).permute(1, 0)
                            for mission_pred, layer in zip(pred, self.normalization_layer)]
        return pred

    def loss_func(self, prediction, label):
        """
        label = [class, n_days, flag]
        :param prediction: prediction = [class, n_days]
        :param labels:
        :return:
        """
        loss = 0.0
        sig = nn.Sigmoid()
        for y_predict, y_true, var, entropy in zip(prediction, label, self.data_var, self.entropy):
            flag = y_true[:, 2]
            ce_loss = self.ce(sig(y_predict[:, 0])*flag, y_true[:, 0]*flag)/entropy
            mse_loss = self.mse(y_true[:, 1]*flag, y_predict[:, 1]*flag)/var

            # normalize the loss
            loss += self.alpha*ce_loss + (1-self.alpha)*mse_loss
            # loss = mse_loss
        return loss

    def fit_transform(self, data_not_emb, data_emb):
        data_not_emb = torch.tensor(data_not_emb.values, device=device).float()
        data_emb = [torch.tensor(X.values, device=device).float() for X in data_emb]
        pred = self.forward(data_not_emb, data_emb)
        return pred

    def evaluate(self, X_train_not_emb, X_train_emb, labels_train, X_val_not_emb, X_val_emb, labels_val):
        labels_options = ["Sur", "GVH2", "GVH3", "CGVH", "DFS"]
        y_train_predict_all = self.fit_transform(X_train_not_emb, X_train_emb)
        y_val_predict_all = self.fit_transform(X_val_not_emb, X_val_emb)

        y_val_true_all = [torch.tensor(label, device=device).float() for label in labels_val]
        y_train_true_all = [torch.tensor(label, device=device).float() for label in labels_train]

        # for each class calculate AUC and mean distance
        auc_val_vec = {}
        auc_train_vec = {}
        for y_train_predict, y_val_predict, y_train_true, y_val_true, name in zip(y_train_predict_all, y_val_predict_all, y_train_true_all, y_val_true_all, labels_options):
            y_train_true = torch.tensor(y_train_true, device=device).float()
            # for each class calculate AUC and mean distance
            y_train_predict = y_train_predict.detach().numpy()
            y_train_true = y_train_true.detach().numpy().astype(int)
            flag_train = y_train_true[:, 2]
            # take flag where there is data
            ind = np.nonzero(flag_train)
            y_train_true_class = y_train_true[ind, 0].squeeze()
            y_train_predict_class = y_train_predict[ind, 0].squeeze()
            y_train_true_days = y_train_true[ind, 1].squeeze()
            y_train_predict_days = y_train_predict[ind, 1].squeeze()

            auc_score_train = roc_auc_score(y_train_true_class, y_train_predict_class)
            diff = abs(y_train_true_days - y_train_predict_days)
            mean_distance_train = np.mean(diff)
            correlation_train = np.corrcoef(y_train_true_days, y_train_predict_days)[0, 1]

            y_val_true = torch.tensor(y_val_true, device=device).float()

            # for each class calculate AUC and mean distance
            y_val_predict = y_val_predict.detach().numpy()
            y_val_true = y_val_true.detach().numpy().astype(int)
            flag_val = y_val_true[:, 2]

            # take flag where there is data
            ind = np.nonzero(flag_val)
            y_val_true_class = y_val_true[ind, 0].squeeze()
            y_val_predict_class = y_val_predict[ind, 0].squeeze()

            y_val_true_days = y_val_true[ind, 1].squeeze()
            y_val_predict_days = y_val_predict[ind, 1].squeeze()

            auc_score_val = roc_auc_score(y_val_true_class, y_val_predict_class)
            diff = abs(y_val_true_days - y_val_predict_days)
            mean_distance_val = np.mean(diff)
            correlation_val = np.corrcoef(y_val_true_days.squeeze(), y_val_predict_days.squeeze())[0, 1]
            print(f"Label {name}\n"
                  f"Train:\n"
                  f"AUC: {auc_score_train:.2f}, MAE: {mean_distance_train:.2f}, Correlation: {correlation_train:.2f}\n"
                  f"Val:\n"
                  f"AUC: {auc_score_val:.2f}, MAE: {mean_distance_val:.2f}, Correlation: {correlation_val:.2f}")

            auc_val_vec[name] = auc_score_val
            auc_train_vec[name] = auc_score_train
            # fpr, tpr, _ = roc_curve(y_true[:, 0]*flag,  y_predict[:, 0]*flag)
            # auc = roc_auc_score(y_true[:, 0]*flag, y_predict[:, 0]*flag)
            # plt.plot(fpr, tpr, label=f"auc={auc:.2f}")
            # plt.legend()
            # plt.title(f"{name}")
            # plt.show()
        return auc_val_vec, auc_train_vec



def train_model(X_train, y_train, X_val, y_val, model_params):
    y_train = [y.values for y in y_train]
    y_val = [y.values for y in y_val]

    # split to embedding features and other features
    # embedding_cols = [col for col in X_train.columns if "emb" in str(col)]
    # X_train_emb = X_train[embedding_cols]
    # X_val_emb = X_val[embedding_cols]
    # X_train = X_train.drop(embedding_cols, axis=1)
    # X_val = X_val.drop(embedding_cols, axis=1)

    # to onehot
    # ntypes_embedding = [len(np.unique(X_train_emb[col])) for col in X_train_emb]
    # X_train_emb = [pd.get_dummies(X_train_emb[col]) for col in X_train_emb.columns]
    # X_val_emb = [pd.get_dummies(X_val_emb[col]) for col in X_val_emb.columns]

    # X_train_emb = [df for df in X_train if "emb" in df.columns[0]]
    # X_val_emb = [df for df in X_val if "emb" in df.columns[0]]
    # X_train_not_emb = pd.concat([df for df in X_train if "emb" not in df.columns[0]], axis=1)
    # X_val_not_emb = pd.concat([df for df in X_val if "emb" not in df.columns[0]], axis=1)

    X_train_emb = X_train[1:]
    X_val_emb = X_val[1:]
    X_train_not_emb = X_train[0]
    X_val_not_emb = X_val[0]
    ntypes_embedding = [len(X.columns) for X in X_train_emb]
    # calc var and entropy
    entrop_vec = [preProcessing.entropy1(labels[:, 0]) for labels in y_train]
    variance_vec = [np.var(labels[:, 1]) for labels in y_train]

    # build datasets
    X_train_dataset = Dataset(X_train_not_emb, X_train_emb, y_train)
    X_val_dataset = Dataset(X_val_not_emb, X_val_emb, y_val)
    train_loader = torch.utils.data.DataLoader(X_train_dataset, batch_size=model_params["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(X_val_dataset, batch_size=X_val.__len__(), shuffle=False)
    model = Model(entrop=entrop_vec, var=variance_vec, n_labels=len(y_train), emb_len=ntypes_embedding,
                  input_shape=X_train_not_emb.shape[1], hidden1=model_params["hidden1"],
                  hidden2=model_params["hidden2"], params=model_params).to(device).float()
    optimizer = optim.Adam(params=model.parameters(), lr=model_params["lr"],  weight_decay=model_params["weight_decay"])
    loss_train = []
    loss_val = []
    epoch = 0
    while not is_converge(loss_val, model_params["MAX_EPOCHS"]):
        epoch += 1
        train_or_test_model(
            model,
            optimizer,
            train_loader,
            loss_train,
            TrainOrTest="Train",
        )
        train_or_test_model(
            model,
            optimizer,
            val_loader,
            loss_val,
            TrainOrTest="Test",
        )
        print(
            f"Epoch: {epoch}, Loss Train: {loss_train[-1]:.3f}, Loss Valid {loss_val[-1]:.3f}"
        )
    print("Evaluation")
    auc_val_vec, auc_train_vec = model.evaluate(X_train_not_emb, X_train_emb, y_train, X_val_not_emb, X_val_emb, y_val)
    return auc_val_vec, auc_train_vec


def train_or_test_model(
    model, optimizer, dataloader, loss_vec, TrainOrTest
):
    if TrainOrTest == "Train":
        model.train()
        optimizer.zero_grad()
        run_model(
            model,
            optimizer,
            dataloader,
            loss_vec,
            TrainOrTest,
        )
    else:
        model.eval()
        with torch.no_grad():
            run_model(
                model,
                optimizer,
                dataloader,
                loss_vec,
                TrainOrTest,
            )


def run_model(
    model, optimizer, dataloader, loss_vec, TrainOrTest
):
    running_loss = 0.0
    for batch in dataloader:
        X, X_to_emb, y = batch
        prediction = model(X, X_to_emb)
        loss = model.loss_func(prediction, y)

        if TrainOrTest == "Train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
    loss_vec.append(running_loss / len(dataloader))


def is_converge(vector, MAX_EPOCHS):
    if len(vector) > MAX_EPOCHS:
        return True
    if len(vector) < 10:
        return False
    if min(vector) < min(vector[-10:]):
        return True
    else:
        return False





