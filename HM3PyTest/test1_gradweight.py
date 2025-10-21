import torch
import torch.nn.functional as F


input = torch.tensor(
    [[ 0.9, 0.8, 0.5 ],
     [ 0.8, 0.2, 0.6 ],
     [ 0.1, 0.4, 0.4 ],
     [ 0.5, 0.5, 0.2 ],
     [ 0.2, 0.8, 0.8 ]],
dtype=torch.float32, requires_grad=True)

weight = torch.tensor(
    [[ 0.9, 0.8 ],
     [ 0.5, 0.8 ],
     [ 0.2, 0.6 ]],
dtype=torch.float32, requires_grad=True)

bias = torch.tensor(
    [0.9, 0.8],
dtype=torch.float32, requires_grad=True)

output = F.linear(input, weight.t(), bias)

grad_output = torch.tensor(
    [[ 0.9, 0.8 ],
     [ 0.5, 0.8 ],
     [ 0.2, 0.6 ],
     [ 0.1, 0.4 ],
     [ 0.4, 0.5 ]],
dtype=torch.float32)

output.backward(grad_output)

# print(input.grad.numpy().round(2))
print(weight.grad.numpy().round(2))
# print(bias.grad.numpy().round(2))