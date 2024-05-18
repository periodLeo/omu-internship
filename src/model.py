# Our modules
from pbc2dataset import PBC2DataSet

# Extern modules
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class BaseLineClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BaseLineClassifier, self).__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc     = nn.Linear(hidden_size, num_classes)
    
    def forward(self, time, series):
        out, _  = self.lstm(series)
        out     = out[:, -1, :]
        out     = self.fc(out)
        return out

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (time, series), labels in dataloader:
            outputs = model(time, series)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    csvfilename = "./data/pbc2.csv"
    data = PBC2DataSet(csvfilename= csvfilename)
    data.remove_nan()

    dataloader = DataLoader(data, batch_size=5, shuffle=True)

    # ~~~~ Define Settings ~~~~
    input_size  = 2
    hidden_size = 16
    num_layers  = 2
    num_classes = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~ Define Hyperparameters ~~~~
    num_epochs      = 20
    learning_rate   = 0.001
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Instantiate model
    model = BaseLineClassifier(input_size, hidden_size, num_layers, num_classes)

    # ~~~~ Learning Phase ~~~~~ Make this a function later
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (time, series), labels in dataloader:
            outputs = model(time, series)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    evaluate_model(model, dataloader)