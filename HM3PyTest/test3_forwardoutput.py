import torch
import numpy as np

# 输入数据
input = torch.tensor([[[[ 0.9, 0.8, 0.5, 0.8 ],
                        [ 0.2, 0.6, 0.1, 0.4 ],
                        [ 0.4, 0.5, 0.5, 0.2 ],
                        [ 0.2, 0.8, 0.8, 0.3 ]],
                       [[ 0.5, 0.1, 0.6, 0.1 ],
                        [ 0.2, 0.9, 0.7, 0.3 ],
                        [ 0, 0.7, 0.7, 0.2 ],
                        [ 0, 0.5, 0.9, 0.8 ]],
                       [[ 0.2, 0.3, 0.2, 0 ],
                        [ 0.8, 0.7, 0.6, 0.8 ],
                        [ 0, 0.7, 0.2, 0.4 ],
                        [ 0.7, 0.1, 0.8, 0.9 ]]]], dtype=torch.float32, requires_grad=True)

# 前向 maxpool
output, indices = torch.nn.functional.max_pool2d(input, kernel_size=2, stride=2, return_indices=True)
print(output.detach().numpy().round(2))

# print("\nPyTorch mask(indices):\n", indices.detach().numpy())
# print("Diff with CUDA mask:\n", indices.detach().numpy() - np.array(
#     [[[[ 0, 3 ],
#        [ 13, 14 ]],
#       [[ 21, 22 ],
#        [ 25, 30 ]],
#       [[ 36, 39 ],
#        [ 41, 47 ]]]]
# ))

# # 反向 maxpool
# grad_output = torch.tensor([[[[ 0.9, 0.8 ],
#                               [ 0.5, 0.8 ]],
#                              [[ 0.2, 0.6 ],
#                               [ 0.1, 0.4 ]],
#                              [[ 0.4, 0.5 ],
#                               [ 0.5, 0.2 ]]]], dtype=torch.float32)
# output.backward(grad_output)
# print("\nPyTorch grad_input:\n", input.grad.detach().numpy().round(2))
# print("Diff with CUDA grad_input:\n", input.grad.detach().numpy().round(2) - np.array(
#     [[[[ 0.9, 0, 0, 0.8 ],
#        [ 0, 0, 0, 0 ],
#        [ 0, 0, 0, 0 ],
#        [ 0, 0.5, 0.8, 0 ]],
#       [[ 0, 0, 0, 0 ],
#        [ 0, 0.2, 0.6, 0 ],
#        [ 0, 0.1, 0, 0 ],
#        [ 0, 0, 0.4, 0 ]],
#       [[ 0, 0, 0, 0 ],
#        [ 0.4, 0, 0, 0.5 ],
#        [ 0, 0.5, 0, 0 ],
#        [ 0, 0, 0, 0.2 ]]]]
# ))