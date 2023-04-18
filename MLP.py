import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(200, 275)
        self.fc2 = nn.Linear(275, 100)
        self.fc3 = nn.Linear(100, 1)

        self.dropout1 = nn.Dropout(p=0.15)
        self.dropout2 = nn.Dropout(p=0.15)

        self.srelu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.srelu(self.dropout1(self.fc1(x)))
        x = self.srelu(self.dropout2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
df=pd.read_csv('c13k_selections.csv')
for i in range(10):
    train=df[df['TEST_TRAIN'+str(i)]=='TRAIN']
    test=df[df['TEST_TRAIN'+str(i)]=='TEST']
    # Create dataset and dataloader
    # You'll need to replace X_train and y_train with your own data
    X_train = train.drop(columns=['Problem','TEST_TRAIN0','TEST_TRAIN1','TEST_TRAIN2','TEST_TRAIN3','TEST_TRAIN4','TEST_TRAIN5','TEST_TRAIN6','TEST_TRAIN7','TEST_TRAIN8','TEST_TRAIN9','B_rate'])
    y_train = train['B_rate']
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Initialize model and optimizer
    model = MLP()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    # Define loss function
    criterion = nn.MSELoss()
    # Train model
    for epoch in range(60):
        epoch_loss = 0.0
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("Epoch: {}, Loss: {}".format(epoch + 1, epoch_loss / (len(dataset) / 128)))
    X_test = test.drop(columns=['Problem','TEST_TRAIN0','TEST_TRAIN1','TEST_TRAIN2','TEST_TRAIN3','TEST_TRAIN4','TEST_TRAIN5','TEST_TRAIN6','TEST_TRAIN7','TEST_TRAIN8','TEST_TRAIN9','B_rate'])
    y_test = test['B_rate']
    print(criterion(model(X_test), y_test))
