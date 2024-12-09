import torch
import torch.nn as nn
import json

from nltk_utils import *

with open('data.json', 'r') as f:
    data = json.load(f)

all_words = []
tags = []
xy = []
bag = []
ignore_words = ['!', '?', '.', ',']

for datum in data:
    tag = datum['tag']
    tags.append(tag)

    for pattern in datum['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set(stem(lower([w for w in all_words if w not in ignore_words]))))
tags = sorted(set(lower(tags)))

X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class NN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        apply_order = [self.l1, self.relu, self.l2, self.relu, self.l3]

        for f in apply_order:
            x = f(x)

        return x


BATCH_SIZE = 8
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(tags)
INPUT_SIZE = len(X_train[0])
LEARNING_RATE = 0.001
EPOCHS = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device:', device)

model = NN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
