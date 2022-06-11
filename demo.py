from model import SentimentModel
import torch
from transformers import BertTokenizerFast
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = SentimentModel().to('cuda')
print("Loading Model...")
model.load_state_dict(torch.load('./checkpoint/model1.pt'))
model.eval()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
classes = {0: 'negative', 1: 'positive'}

print("Demo start...")
while True :
    input_utterance = str(input('Input utterance: '))
    if input_utterance == 'exit' :
        break

    x = tokenizer(input_utterance, max_length=200, padding='max_length', truncation=True, return_tensors="pt")
    x = {key: val.to('cuda') for key, val in x.items()}
    with torch.no_grad():
        outputs = model(x)
        _, predicts = torch.max(outputs.data, 1)
        
    print("[{}]".format(classes[predicts.item()]), end='\n\n')
