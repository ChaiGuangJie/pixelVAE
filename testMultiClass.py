import torch
import torch.nn as nn
from torch.nn import functional as F

class B(nn.Module):
    def __init__(self):
        super().__init__()
        print('B')

        self.fc1 = nn.Linear(5,8)
        self.fc2 = nn.Linear(8,10)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(h))

class A(nn.Module):
    def __init__(self):
        super().__init__()
        print('A')
        self.sub = B()
        self.fc1 = nn.Linear(10,20)
        self.fc2 = nn.Linear(20,30)

    def forward(self, x):
        # h = F.relu(self.fc1(x))
        # return F.sigmoid(self.fc2(h))
        h1 = self.sub.forward(x)
        h2 = F.relu(self.fc1(h1))
        return F.sigmoid(self.fc2(h2))

# class C(A, B):
#     def __init__(self):
#         super().__init__()
#         print('C')





model = A()
for param in model.parameters():
    print(type(param.data), param.size())
print('end')