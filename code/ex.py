import torch

t = torch.tensor([[0,0,1], [0,1,0], [1,0,0]])
x =  torch.tensor([[0,0,1], [0,0,0], [0,1,0]])
print(torch.sum(torch.all(t == x, dim=1)).item())

print(t==x)
