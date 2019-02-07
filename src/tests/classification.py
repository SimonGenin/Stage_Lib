import torch
import torch.nn as nn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisModel(nn.Module):

    def __init__(self, N):
        super(IrisModel, self).__init__()
        self.linear_1 = nn.Linear(N, 5)
        self.linear_2 = nn.Linear(5, 5)
        self.linear_3 = nn.Linear(5, N)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y_pred = self.linear_1(x)

        y_pred = self.linear_2(y_pred)

        y_pred = self.linear_3(y_pred)

        # y_pred = self.softmax(y_pred)

        return y_pred


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.33, random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train)

model = IrisModel(4)

epoch = 30000

loss = None

optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for current_epoch in range(epoch):

    y_pred = model(X_train)

    loss = criterion(y_pred, y_train)

    if current_epoch % 1000 == 0:
        print("Loss at ", current_epoch, "is", loss.item())

    optim.zero_grad()
    loss.backward()
    optim.step()

accuracy = 0
for index, test_item in enumerate(X_test):

    y_pred = model(torch.from_numpy(test_item).float())

    y_pred = torch.nn.functional.softmax(y_pred, dim=0)

    arg = torch.argmax(y_pred)

    print('has', y_pred)
    print('for', y_test[index])

    if (arg == y_test[index]):
        accuracy += 1

accuracy /= len(X_test)


print("Accuracy is", accuracy)
