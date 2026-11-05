import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms


batch_size = 256
device = torch.device('cpu')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=batch_size,shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),])),
        batch_size=batch_size,shuffle=True
)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)
    
    def forward(self,x):
        return self.l2(F.relu(self.l1(x)))
    
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(),lr=1e-2)

epochs = 15
for epochs in range(epochs):
    #print(f"epochs:{epochs}")
    model.train()
    #print("start train")
    train_loss = 0
    for batch_idx,(x,y) in enumerate(train_loader):
        x = x.view(x.shape[0],-1).to(device)
        y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output,y)
        train_loss += loss
        loss.backward()
        optimizer.step()
      
    print(f"epochs:{epochs},train_loss={(train_loss/(batch_idx+1)):.4f}")
        
    model.eval()
    #print("start test")
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x,y in test_loader:
            x = x.view(x.shape[0],-1).to(device)
            y.to(device)
            output = model(x) 
            test_loss += F.cross_entropy(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
        
        test_loss = test_loss/(batch_idx+1)
        acc = correct/len(test_loader.dataset)
        print(f"epochs:{epochs},test_loss={test_loss:.4f},acc:{acc:.4f}")