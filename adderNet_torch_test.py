from adderNet_torch import AdderNet
import torch
import torch.nn as nn

model = AdderNet([2, 16, 1])

x_train = torch.empty(100, 2).uniform_(-10, 10)
y_train = (x_train[:, 0]**2 + x_train[:, 1]**2).reshape(-1, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

prev_loss = 0
i = 0
while True:
    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(i, loss.item())
        if (abs(loss.item()-prev_loss) < 0.05):
            break
        prev_loss = loss.item()

    i += 1
        

    

torch.save(model.state_dict(), "addernet_weights.pth")