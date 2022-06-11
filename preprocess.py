import csv
import pickle
from transformers import BertTokenizerFast

# Preprocess Train Data
with open('./data/train.csv', newline='') as f:
	reader = csv.reader(f)
	data = list(reader)

x_train = [ d[2] for d in data ]
y_train = [ d[0] for d in data ]

with open('./data/x_train', 'wb') as fp:
	pickle.dump(x_train, fp)

with open('./data/y_train', 'wb') as fp:
	pickle.dump(y_train, fp)

print("Finished preprocess train data")

# Preprocess Test Data
with open('./data/test.csv', newline='') as f:
	reader = csv.reader(f)
	data = list(reader)

x_test = [ d[2] for d in data ]
y_test = [ d[0] for d in data ]

with open('./data/x_test', 'wb') as fp:
	pickle.dump(x_test, fp)

with open('./data/y_test', 'wb') as fp:
	pickle.dump(y_test, fp)

print("Finished preprocess test data")

