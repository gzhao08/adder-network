from adderNet_torch import AdderNet
import torch
import torch.nn as nn

x_test = torch.empty(1000, 2).uniform_(-10, 10)
y_test = (x_test[:, 0]**2 + x_test[:, 1]**2).reshape(-1, 1)

loss_fn = nn.MSELoss()

def quantize_tensor_symmetric(x, num_bits=8):
    qmax = 2**(num_bits - 1) - 1
    qmin = -2**(num_bits - 1)

    scale = x.abs().max() / qmax

    if scale == 0:
        return x.clone(), scale

    x_int = torch.round(x / scale)
    x_int = torch.clamp(x_int, qmin, qmax)

    x_quant = x_int * scale
    return x_quant, scale

def quantize_model_weights(model, num_bits=8):
    quantized_state = {}

    for name, param in model.state_dict().items():
        q_param, scale = quantize_tensor_symmetric(param, num_bits)
        quantized_state[name] = q_param
        # print(name, "scale =", scale.item() if torch.is_tensor(scale) else scale)

    model.load_state_dict(quantized_state)
    return model

for bits in [16, 12, 8, 6, 4, 3, 2]:
    model_q = AdderNet([2, 16, 1])
    model_q.load_state_dict(torch.load("addernet_weights_2-16-1.pth"))
    model_q = quantize_model_weights(model_q, num_bits=bits)
    model_q.eval()

    with torch.no_grad():
        pred = model_q(x_test)
        loss = loss_fn(pred, y_test)

    print(f"Bits: {bits}")
    # print(f"Output: {pred}")
    print(f"Loss: {loss.item()}")
    print()



