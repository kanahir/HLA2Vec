import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
MAX_EPOCHS = 100
lr = 0.001

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, device=device).float()

    # Length of the Dataset
    def __len__(self):
        return self.data.shape[0]

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        return self.data[idx, :]


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, n_categories, enc_hideen=4, latent_size=4, dec_hidden=4):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=enc_hideen
        )
        self.encoder_output_layer = nn.Linear(
            in_features=enc_hideen, out_features=latent_size
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=latent_size, out_features=dec_hidden
        )
        self.decoder_output_layer = nn.Linear(
            in_features=dec_hidden, out_features=input_shape
        )
        self.n_categories = n_categories
        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed

    def loss_func(self, input, prediction):
        # split the prediction
        split_pred = torch.split(prediction, self.n_categories)
        # calculate softmax on the predictions
        split_pred = list(map(lambda x: F.softmax(x), split_pred))
        prediction = torch.concat(split_pred)
        return self.loss(input, prediction)

    def fit_transform(self, data):
        data = torch.tensor(data.values, device=device, dtype=torch.float32)
        activation = self.encoder_hidden_layer(data)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        return code


def train_model(data, n_categories):
    n_features = data.shape[1]
    # train_test_split
    X_train, X_test = data_functions.train_test_split(data)
    X_train = Dataset(X_train)
    X_test = Dataset(X_test)
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=X_test.__len__(), shuffle=False)
    model = AutoEncoder(input_shape=n_features, n_categories=n_categories).to(device).float()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    loss_train = []
    loss_test = []
    epoch = 0
    while not is_converge(loss_test):
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
            test_loader,
            loss_test,
            TrainOrTest="Test",
        )
        print(
            f"Epoch: {epoch}, Loss Train: {loss_train[-1]:.3f}, Loss Valid {loss_train[-1]:.3f}"
        )

    return model


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
        prediction = model(batch)
        loss = model.loss(batch, prediction)

        if TrainOrTest == "Train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        loss_vec.append(running_loss / len(dataloader))


def is_converge(vector, MAX_EPOCHS=MAX_EPOCHS):
    if len(vector) > MAX_EPOCHS:
        return True
    if len(vector) < 10:
        return False
    if min(vector) < min(vector[-10:]):
        return True
    else:
        return False





