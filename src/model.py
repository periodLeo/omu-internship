# Our modules
from pbc2dataset import PBC2DataSet
from utils import collate_fn_pre_padding

# Extern modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from time import time

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMBinaryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class LinearBinaryClassifier(nn.Module):
    def __init__(self, num_series, series_length):
        """
        Initialize the LinearClassifier model.

        Parameters:
        num_series (int): Number of parallel time series (features).
        series_length (int): Length of each time series.
        """
        super(LinearClassifier, self).__init__()
        self.num_series = num_series
        self.series_length = series_length

        # The input features are flattened time series data
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_series * series_length, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_series, series_length).
                          Each entry in the batch is a multivariate time series.

        Returns:
        torch.Tensor: The output predictions of shape (batch_size, num_classes).
        """
        # Flatten the time series data: from (batch_size, num_series, series_length)
        # to (batch_size, num_series * series_length)
        x = self.flatten(x)

        # Apply the linear layer to get logits for each class
        logits = self.linear(x)

        return logits

def train_model(model: nn.Module, dataloader, num_epochs: int=20) -> None:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_tab = []
    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in dataloader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss_tab.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1)%10 == 0:
            loss_mean = sum(loss_tab)/len(loss_tab)
            loss_tab = []
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            evaluate_model_acc(model, dataloader)

def evaluate_model_auc(model: nn.Module, dataloader) -> float:
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences)
            # Assuming the model outputs logits, you will need to apply a sigmoid function
            # to get probabilities if you're dealing with binary classification.
            
            all_labels.extend(labels.tolist())
            all_predictions.extend(outputs.tolist())

    # Compute AUC score
    auc_score = roc_auc_score(all_labels, all_predictions)
    print(f'Test AUC: {auc_score:.4f}')
    return auc_score

def evaluate_model_acc(model: nn.Module, dataloader) -> float:
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in dataloader:        
            outputs = model(sequences)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct/total
        print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

if __name__ == "__main__":
    # Load base dataset
    csvfilename = "./data/pbc2.csv"
    data = PBC2DataSet(csvfilename= csvfilename)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, random_state=23)

    # Rebuild dataset from splitted data to pass them in dataloader
    data_train = PBC2DataSet(tensor_data=(X_train, y_train))
    data_test  = PBC2DataSet(tensor_data=(X_test, y_test))

    # Create dataloader
    train_dataloader = DataLoader(data_train, batch_size=16, shuffle=True, collate_fn=collate_fn_pre_padding)
    test_dataloader  = DataLoader(data_test, batch_size=16, shuffle=True, collate_fn=collate_fn_pre_padding)
    dataloader       = DataLoader(data, batch_size=16, shuffle=True, collate_fn=collate_fn_pre_padding)

    # ~~~~ Define Settings ~~~~
    input_size  = data_train.X[0].shape[1]
    hidden_size = 40
    num_layers  = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~ Define Hyperparameters ~~~~
    num_epochs      = 100
    learning_rate   = 0.0001
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Instantiate model
    model = LSTMBinaryClassifier(input_size, hidden_size, num_layers)
    train_model(model, train_dataloader, num_epochs=num_epochs)
    evaluate_model_auc(model, test_dataloader)

    # for i in range(25):
    #     num_epochs = 50 + i*10
    #     print(f"Start for epchos = {num_epochs}")
    #     start_time = time()

    #     model = LSTMBinaryClassifier(input_size, hidden_size, num_layers)
    #     train_model(model, train_dataloader, num_epochs=num_epochs)
    #     accu = evaluate_model(model, test_dataloader)
    #     time_to_run = time() - start_time
    #     print(f"End in {time_to_run} s")

    #     epochs_tab.append(num_epochs)
    #     time_tab.append(time_to_run)
    #     accuracy_tab.append(accu)

    # fig, ax1 = plt.subplots()

    # # Plot times_for_run on the primary y-axis
    # color = 'tab:blue'
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Time for Run (s)', color=color)
    # ax1.plot(epochs_tab, time_tab, color=color, label='Time for Run')
    # ax1.tick_params(axis='y', labelcolor=color)

    # # Create a second y-axis to plot accuracy_tab
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:red'
    # ax2.set_ylabel('Accuracy', color=color)
    # ax2.plot(epochs_tab, accuracy_tab, color=color, label='Accuracy')
    # ax2.tick_params(axis='y', labelcolor=color)

    # # Add a title and a legend
    # fig.suptitle('Time for Run and Accuracy over Epochs')
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # # Display the plot
    # plt.show()