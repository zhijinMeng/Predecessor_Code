import numpy as np
import torch
# a = torch.randn(2, 3)
# a = torch.IntTensor(5)
a = torch.tensor(5)
# print(a.shape)
# print(len(a.shape))
# print(a.type())
# print(type(a))
# print(isinstance(a, torch.IntTensor))

# import from numpy to tensor
a = np.array([1, 2, 3])
# print(torch.from_numpy(a))


# import from list to tensor
a = torch.tensor([1, 2, 3])
# print(a)


# uninitialized tensor
a = torch.empty(2, 3)
# print(a)

# rand/rand_like, randint
a = torch.rand(2, 3)
# print(a)

a = torch.rand_like(a)
# print(a)

# normal & std
a = torch.normal(mean=torch.full([10], 0.), std=torch.arange(0.1, 0, -0.1))
# print(a)

# Indexing & Slicing
a = torch.rand(4, 3, 28, 28)
print(a[:1, :, :, :])
print(a[:1, 1:, 1, :])