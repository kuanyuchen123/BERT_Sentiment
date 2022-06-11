import torch
import pickle
from model import SentimentData, SentimentModel
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print("Loading train data...")
with open ('./data/x_train', 'rb') as fp:
    x_train = pickle.load(fp)

with open ('./data/y_train', 'rb') as fp:
    y_train = pickle.load(fp)

num_epoch = 10
batch_size = 64
lr = 2e-5

train_data = SentimentData(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model = SentimentModel().to('cuda')
model.train()

critierion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
steps = math.ceil(len(y_train)/batch_size)

print("Start training...")
for e in range(num_epoch):
    total = 0
    correct = 0
    running_loss = 0

    for step, (input,label) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input)
        loss = critierion(outputs, label)
        loss.backward()
        optimizer.step()

        total += label.size(0)
        _, predicts = torch.max(outputs.data, 1)

        correct += (predicts == label).sum().item()
        running_loss += loss.data.item()

        if (step+1) % 30 == 0 :
            print("[Epoch {}, Step {}] Loss: {:.4f}, Acc: {:.4f}".format(e+1, step+1, running_loss/(step+1), correct/total))


    print("[Epoch {}] Loss: {:.4f}, Acc: {:.4f}".format(e+1, running_loss/steps, correct/total))
    total = 0
    running_loss = 0
    correct = 0
    torch.save(model.state_dict(), "./checkpoint/model{}.pt".format(e+1))

print('Finished training')
