from torch.utils.data import Dataset
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizerFast

class SentimentData(Dataset):
    def __init__(self, x, y):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        tokenized_x = self.tokenizer(self.x[idx], max_length=200, padding='max_length', truncation=True, return_tensors="pt")
        x = {key: val[0].to('cuda') for key, val in tokenized_x.items()}
        y = torch.tensor(int(self.y[idx])-1, device='cuda')
        return x,y

    def __len__(self):
        return len(self.y)


class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 2)
        
    def forward(self, input):
        outputs = self.bert(**input)
        logits = self.linear(outputs.last_hidden_state[:,0,:])
        return logits

