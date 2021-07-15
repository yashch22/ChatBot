import json
from nltk_utility import tokenize, stem, bag_of_words
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader
from model import NN
#Pipeline for creating training data
# sentence --> tokenize --> lower_case + stemming --> remove punctuations --> apply bag of words

with open('intents.json','r') as f:
    intents = json.load(f)



vocabulary = []
tags = []
data = []


for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        vocabulary.extend(words)
        data.append((words,tag))
     


not_to_include = ['?','!','.',',']
vocabulary = [stem(w) for w in vocabulary if w not in not_to_include ]
vocabulary = sorted(set(vocabulary))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (pattern_sentence, tag) in data:
    bow = bag_of_words(pattern_sentence,vocabulary) 
    X_train.append(bow)
    label = tags.index(tag)
    Y_train.append(label)


X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #Get dataset[idx]
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples 
    


# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs =  1000


dataset = ChatDataset()
train_loader = Dataloader(dataset = dataset,batch_size = batch_size,shuffle =True,num_workers=2)

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NN(input_size,hidden_size,output_size)

#Loss and optimizer
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters, lr = learning_rate)


#Training phase
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #fwd pass
        outputs = model(words)
        loss = loss_criterion(outputs, labels)

        #optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print every 100th loss
    if (epoch+1)%100==0:
         print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss = {loss.item():.4f}')


#Saving model
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": vocabulary,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

