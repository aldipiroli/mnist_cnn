import torch 
import torch.nn as nn

out = torch.randn(5, 3,2 )

print("Before", out.shape)
# out = out.reshape(out.size(0), -1)
# out = torch.flatten(out)

_, max_ = torch.max(out, 1)
print("After", max_)
