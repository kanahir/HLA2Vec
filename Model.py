from torch import nn
import torch
import numpy as np


class NeuralNetwork(nn.Module):

    def __init__(self, dims, num_output, dropout_prob=0.0, initialization=False, bin_pred=True):
        super(NeuralNetwork, self).__init__()

        lin_fc = []
        for i in range(len(dims) - 2):
            lin_fc.append(nn.Linear(dims[i], dims[i+1]))
        self.lin_fc = nn.ModuleList(lin_fc)

        bin_fc = []
        for _ in range(num_output):
            bin_fc.append(nn.Linear(dims[-2], dims[-1]))
        self.bin_fc = nn.ModuleList(bin_fc)

        cont_fc = []
        for _ in range(num_output):
            cont_fc.append(nn.Linear(dims[-2], dims[-1]))
        self.cont_fc = nn.ModuleList(cont_fc)

        self.activate = nn.ReLU()
        self.drop = nn.Dropout(dropout_prob)
        self.sigm = nn.Sigmoid()
        self.binary_predict = bin_pred

        if initialization:
            for fc1 in self.lin_fc:
                nn.init.xavier_uniform_(fc1.weight)
            for fc2 in self.bin_fc:
                nn.init.xavier_uniform_(fc2.weight)
            for fc3 in self.cont_fc:
                nn.init.xavier_uniform_(fc3.weight)

    def forward(self, x):
        for fc in self.lin_fc:
            x = self.activate(fc(x))
            x = self.drop(x)

        output = []
        for i in range(len(self.cont_fc)):
            y_bin = self.sigm(self.bin_fc[i](x))
            # y_bin = self.bin_fc[i](x)
            y_cont = self.cont_fc[i](x)
            output.append(y_bin)
            output.append(y_cont)

        return output

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            data = torch.Tensor(data)
            pred_bin, pred_cont = self(data)
            if self.binary_predict:
                pred = pred_bin.view(-1)
            else:
                pred = pred_cont.view(-1)
            return np.array(pred)