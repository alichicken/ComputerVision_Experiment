class MLP(nn.Modules):
    def __init__(self):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)
    
    def forward(self,x):
        return self.l2(F.relu(self.l1(x)))
    
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(),lr=1e-2)
