import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('emails_edit.csv', delimiter=',')
X = dataset[:,0:3000]
y = dataset[:,3000]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


input_size = X.shape[1]
hidden_size = 64
output_size = 1
# define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size, input_size)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(input_size, output_size)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = NeuralNetwork()


# train the model
loss_fn   = nn.MSELoss()  # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

#pickle.dump(model, open('model.pkl', 'wb'))
