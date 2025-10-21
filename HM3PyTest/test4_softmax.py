import numpy as np
import torch
import torch.nn.functional as F

input = np.array([
    [0.9, 0.8, 0.5, 0.8],
    [0.2, 0.6, 0.1, 0.4],
    [0.4, 0.5, 0.5, 0.2]
], dtype=np.float32)
target = np.array([3, 3, 1], dtype=np.int64)

# 转为 torch tensor
x = torch.tensor(input, requires_grad=True)
y = torch.tensor(target)

# softmax
softmax_out = F.softmax(x, dim=1)
print(softmax_out.detach().numpy())

# # cross entropy loss
# loss = F.cross_entropy(x, y, reduction='mean')
# print("\nPyTorch cross entropy loss:", loss.item())

# # backward 得到 grad_output
# loss.backward()
# print("\nPyTorch grad_output:\n", x.grad.detach().numpy())

