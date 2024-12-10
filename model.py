import torch
import torch.nn as nn
import json
from numpy import ndarray, dtype

from nltk_utils import *

with open('data.json', 'r') as f:
    data = json.load(f)

all_words: list[str] = []  # List to store all unique words across all patterns
tags: list[str] = []  # List to store unique tags (categories) from the dataset
xy: list[tuple[list[str], str]] = []  # List of tuples (tokenized_sentence, tag) for each pattern in the dataset
bag: ndarray[int, dtype] = []  # Placeholder variable for the bag-of-words representation (used later)
IGNORE_WORDS = ['!', '?', '.', ',']  # Words to ignore during preprocessing

# Process the dataset
for datum in data:
    tag = datum['tag']
    tags.append(tag)  # Add the tag to the list of tags

    for pattern in datum['patterns']:
        w = tokenize(pattern)  # Tokenize the pattern into individual words
        all_words.extend(w)  # Add all words from the pattern to the list of all_words
        xy.append((w, tag))  # Add the tokenized sentence and tag as a tuple to xy

# Preprocess all words: remove punctuation, lowercase, and stem for consistency
all_words = sorted(set(stem(lower([w for w in all_words if w not in IGNORE_WORDS]))))  # Unique and sorted words
tags = sorted(set(lower(tags)))  # Unique and sorted list of tags

# Prepare training data
X_train = []  # Feature vectors (bag of words for each sentence)
Y_train = []  # Labels (index corresponding to the tag)

for (pattern_sentence, tag) in xy:
    # Create a "bag of words" representation for each sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)  # Add the feature vector for the sentence
    label = tags.index(tag)  # Convert tag to its corresponding numerical index
    Y_train.append(label)  # Add the label to the list of labels

# Convert training data to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)


class NN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Define a simple feedforward neural network with 3 layers
        self.l1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Hidden to hidden
        self.l3 = nn.Linear(hidden_size, num_classes)  # Hidden to output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Define the forward pass using a list for clarity
        apply_order = [self.l1, self.relu, self.l2, self.relu, self.l3]

        for f in apply_order:  # Sequentially apply layers and activations
            x = f(x)

        return x


# Hyperparameters and model configuration
BATCH_SIZE = 8
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(tags)  # Number of classes (tags)
INPUT_SIZE = len(X_train[0])  # Feature vector size
LEARNING_RATE = 0.001
EPOCHS = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
print('Running on device:', device)

# Initialize the model and move it to the appropriate device
model = NN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
