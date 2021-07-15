import torch
import torch.nn as nn

class NN(nn.Module):
    super(NN, self).__init__()
    def __init__(self,input_size, hidden_size, num_classes):
        self.l1 = nn.Linear(input_size, hidden_size )
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu =  nn.ReLU()

    def forward(self,x):
        out = self.l1(x)
        out = self.relu()
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        #don't apply activation and softmax as cross entropy will do it.
        return out
