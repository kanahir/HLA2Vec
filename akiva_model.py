import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import preprocessing
import preProcessing
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import scipy
labels_options = ["Sur", "GVH2", "GVH3", "CGVH", "DFS"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch_size = 64
# MAX_EPOCHS = 100
# lr = 0.005
# wd = 0.0015
# drop = 0.3
# alpha = 0.7
# alpha = 0
# hidden1 = 20
# hidden2 = 5


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, device=device).float()
        self.labels = torch.tensor(labels, device=device).float()

    # Length of the Dataset
    def __len__(self):
        return self.data.shape[0]

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx, :]


class Model(nn.Module):
    def __init__(self, entrop, var, input_shape, params):
        super().__init__()
        self.alpha = params["alpha"]
        hidden1 = params["hidden1"]
        hidden2 = params["hidden2"]
        drop = params["dropout"]
        self.var = var
        self.entropy = entrop
        self.drop = nn.Dropout(p=drop)
        self.sigmoid = nn.Sigmoid()
        self.hidden1 = nn.Linear(
            in_features=input_shape, out_features=hidden1
        )
        self.hidden2 = nn.Linear(
            in_features=hidden1, out_features=hidden2
        )

        self.binary_output = nn.Linear(
            in_features=hidden2, out_features=1)

        self.continuous_output = nn.Linear(
            in_features=hidden2, out_features=1)

        self.ce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.drop(torch.relu(x))
        x = self.hidden2(x)
        x = self.drop(torch.relu(x))
        # last layer
        binary_pred = self.sigmoid(self.binary_output(x))
        continuous_pred = self.continuous_output(x)
        pred = torch.squeeze(torch.stack([binary_pred, continuous_pred], dim=1))
        return pred

    def loss_func(self, y_predict, y_true):
        """
        label = [class, n_days, flag]
        :param prediction: prediction = [class, n_days]
        :param labels:
        :return:
        """
        flag = y_true[:, 2]
        ce_loss = self.ce(y_predict[:, 0]*flag, y_true[:, 0]*flag)
        mse_loss = self.mse(y_true[:, 1]*flag, y_predict[:, 1]*flag)/self.var

        # normalize the loss
        loss = self.alpha*ce_loss + (1-self.alpha)*mse_loss
        # loss = mse_loss
        return loss

    def fit_transform(self, data):
        data = torch.tensor(data, device=device).float()
        pred = self.forward(data)
        return pred

    def evaluate(self, data_train, labels_train, data_val, labels_val, label_name):
        y_train_predict = self.fit_transform(data_train)
        y_train_true = torch.tensor(labels_train, device=device).float()
        # for each class calculate AUC and mean distance
        y_train_predict = y_train_predict.detach().numpy()
        y_train_true = y_train_true.detach().numpy().astype(int)
        flag_train = y_train_true[:, 2]
        # take flag where there is data
        ind = np.nonzero(flag_train)
        y_train_true_class = y_train_true[ind, 0]
        y_train_predict_class = y_train_predict[ind, 0]
        y_train_true_days = y_train_true[ind, 1]
        y_train_predict_days = y_train_predict[ind, 1]

        auc_score_train = roc_auc_score(y_train_true_class.transpose(), y_train_predict_class.transpose())
        diff = abs(y_train_true_days - y_train_predict_days)
        mean_distance_train = np.mean(diff)
        correlation_train = np.corrcoef(y_train_true_days, y_train_predict_days)[0, 1]

        y_val_predict = self.fit_transform(data_val)
        y_val_true = torch.tensor(labels_val, device=device).float()

        # for each class calculate AUC and mean distance
        y_val_predict = y_val_predict.detach().numpy()
        y_val_true = y_val_true.detach().numpy().astype(int)
        flag_val = y_val_true[:, 2]

        # take flag where there is data
        ind = np.nonzero(flag_val)
        y_val_true_class = y_val_true[ind, 0]
        y_val_predict_class = y_val_predict[ind, 0]

        y_val_true_days = y_val_true[ind, 1]
        y_val_predict_days = y_val_predict[ind, 1]


        auc_score_val = roc_auc_score(y_val_true_class.squeeze(), y_val_predict_class.squeeze())
        # calc correlation

        diff = abs(y_val_true_days - y_val_predict_days)
        mean_distance_val = np.mean(diff)
        correlation_val = scipy.stats.spearmanr(y_val_true_class.squeeze(), y_val_predict_class.squeeze())[0]
        # correlation_val = np.corrcoef(y_val_true_days.squeeze(), y_val_predict_days.squeeze())[0, 1]
        print(f"Label {label_name}\n"
              f"Train:\n"
              f"AUC: {auc_score_train:.2f}, MAE: {mean_distance_train:.2f}, Correlation: {correlation_train:.2f}\n"
              f"Val:\n"
              f"AUC: {auc_score_val:.2f}, MAE: {mean_distance_val:.2f}, Correlation: {correlation_val:.2f}")
        return auc_score_val, auc_score_train, correlation_val



        # plot
        # fpr, tpr, _ = roc_curve(y_train_true[:, 0].squeeze(),  y_train_predict[:, 0].squeeze())
        # auc = roc_auc_score(y_train_true[:, 0].squeeze(), y_train_predict[:, 0].squeeze())
        # plt.plot(fpr, tpr, label=f"Train auc={auc:.2f}", color="b")
        #
        # fpr, tpr, _ = roc_curve(y_val_true[:, 0].squeeze(),  y_val_predict[:, 0].squeeze())
        # auc = roc_auc_score(y_val_true[:, 0].squeeze(), y_val_predict[:, 0].squeeze())
        # plt.plot(fpr, tpr, label=f"Val auc={auc:.2f}", color="orange")
        #
        # plt.legend()
        # plt.title(f"{label_name}")
        # plt.savefig(f"plots/my_data_{label_name}.png")
        # plt.savefig(f"plots/akiva_data_{label_name}.png")

        # plt.show()


def train_model(X_train, y_train, X_val, y_val, params=None):
    if params is None:
        params = {"MAX_EPOCHS": 300, "batch_size": 64, "lr": 0.000836,
         "weight_decay": 0.003, "dropout": 0.0, "alpha": 0.7,
         "hidden1": 10, "hidden2": 5}
    # normalization
    X_train = X_train.__array__()
    X_val = X_val.__array__()
    scaler = preprocessing.StandardScaler().fit(X_train[:, :2])
    X_train[:, :2] = scaler.transform(X_train[:, :2])
    X_val[:, :2] = scaler.transform(X_val[:, :2])
    y_train = [y.values for y in y_train]
    y_val = [y.values for y in y_val]

    # calc var and entropy
    entrop_vec = [preProcessing.entropy1(labels[:, 0]) for labels in y_train]
    variance_vec = [np.var(labels[:, 1]) for labels in y_train]

    auc_val_vec = {}
    auc_train_vec = {}
    correlation_vec = {}
    models = {label: None for label in labels_options}
    for i, label in enumerate(labels_options):
        print(f"Task {label}")
        # train_test_split
        X_train_dataset = Dataset(X_train, y_train[i])
        X_val_dataset = Dataset(X_val, y_val[i])
        train_loader = torch.utils.data.DataLoader(X_train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(X_val_dataset, batch_size=X_val.__len__(), shuffle=False)

        model = Model(entrop=entrop_vec[i], var=variance_vec[i], input_shape=X_train.shape[1], params=params).to(device).float()
        optimizer = optim.Adam(params=model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        loss_train = []
        loss_val = []
        epoch = 0
        while not is_converge(loss_val, MAX_EPOCHS=params["MAX_EPOCHS"]):
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
        auv_val, auc_train, correlation_val = model.evaluate(X_train, y_train[i], X_val, y_val[i], label)
        auc_val_vec[label] = auv_val
        auc_train_vec[label] = auc_train
        correlation_vec[label] = correlation_val
        models[label] = model
    return auc_val_vec, auc_train_vec, correlation_vec, models


def train_or_test_model(
    model, optimizer, dataloader, loss_vec, TrainOrTest
):
    if TrainOrTest == "Train":
        model.train()
        # optimizer.zero_grad()
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
        X, y = batch
        prediction = model(X)
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





