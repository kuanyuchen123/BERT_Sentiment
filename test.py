import torch
import pickle
from model import SentimentData, SentimentModel
from torch.utils.data import DataLoader
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print("Loading test data...")
with open ('./data/x_test', 'rb') as fp:
    x_test = pickle.load(fp)

with open ('./data/y_test', 'rb') as fp:
    y_test = pickle.load(fp)

test_data = SentimentData(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

model = SentimentModel().to('cuda')
model.load_state_dict(torch.load('./checkpoint/model1.pt'))
model.eval()

print("Start testing...")
print("Total steps: {}".format(len(y_test)/512))
with torch.no_grad():
    total = 0
    correct = 0
    for step, (input,label) in enumerate(test_loader):
        outputs = model(input)
        total += label.size(0)
        _, predicts = torch.max(outputs.data, 1)
        correct += (predicts == label).sum().item()
        if step % 50 == 0 :
            print( "[Step {}] Accuracy: {:.4f}".format(step, correct/total) )

    print("Accuracy: {:.4f}".format(correct/total))



