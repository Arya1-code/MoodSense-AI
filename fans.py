import torch

x = torch.randn(5000, 5000).cuda()
y = torch.randn(5000, 5000).cuda()

z = torch.matmul(x, y)

print(z.shape)
print(torch.cuda.get_device_name(0))