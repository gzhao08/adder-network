from adderNet_torch import AdderNet
import torch
import torch.nn as nn

model = AdderNet([2, 16, 1])
model.load_state_dict(torch.load("addernet_weights.pth"))
model.eval()

x_test = torch.tensor([[2.0, 7.0],[12,-5],[-11,-11]])
pred = model(x_test)
print(pred)