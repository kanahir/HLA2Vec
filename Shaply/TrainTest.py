import torch
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr


def train(dataloader, model, alpha, var, device, optimizer):
    model.train()
    return compute_loss_auc_corr(dataloader, model, alpha, var, device, is_train=True, optimizer=optimizer)


def test(dataloader, model, alpha, var, device):
    model.eval()
    with torch.no_grad():
        return compute_loss_auc_corr(dataloader, model, alpha, var, device)


def compute_loss_auc_corr(dataloader, model, alpha, var, device, is_train=False, optimizer=None):

    cum_loss = 0.0
    cum_pred_bin = list()
    cum_pred_cont = list()
    cum_y_bin = list()
    cum_y_cont = list()
    cum_flag = list()

    for X, y_bin, y_cont, flag in dataloader:
        X, y_bin, y_cont, flag = X.to(device), y_bin.to(device), y_cont.to(device), flag.to(device)

        pred_bin, pred_cont = model(X)
        pred_bin = pred_bin.view(-1)
        pred_cont = pred_cont.view(-1)

        loss_bce = nn.BCELoss(reduction='none')
        # loss_bce = nn.BCEWithLogitsLoss(reduction='none')
        loss_bin = loss_bce(pred_bin, y_bin)

        loss_mse = nn.MSELoss(reduction='none')
        loss_cont = loss_mse(pred_cont, y_cont) / var

        loss = alpha * loss_bin + (1 - alpha) * loss_cont

        loss = torch.sum(torch.multiply(loss, flag))
        cum_loss += loss.item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # sigm = nn.Sigmoid()
        # pred_bin = sigm(pred_bin)
        cum_pred_bin += pred_bin.tolist()
        cum_y_bin += y_bin.tolist()
        cum_pred_cont += pred_cont.tolist()
        cum_y_cont += y_cont.tolist()
        cum_flag += flag.tolist()

    cum_pred_bin = [cum_pred_bin[i] for i in range(len(cum_flag)) if cum_flag[i] != 0]
    cum_y_bin = [cum_y_bin[i] for i in range(len(cum_flag)) if cum_flag[i] != 0]
    cum_pred_cont = [cum_pred_cont[i] for i in range(len(cum_flag)) if cum_flag[i] != 0]
    cum_y_cont = [cum_y_cont[i] for i in range(len(cum_flag)) if cum_flag[i] != 0]

    cum_loss /= sum(cum_flag)
    auc = roc_auc_score(cum_y_bin, cum_pred_bin)
    corr, pval = spearmanr(cum_y_cont, cum_pred_cont)
    # fpr, tpr, thresholds = roc_curve(cum_y_bin, cum_pred_bin)

    return cum_loss, auc, corr # , [fpr, tpr, thresholds]
