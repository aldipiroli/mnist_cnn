import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


x = np.random.randn(2)
print(x)

plt.plot(np.arange(len(x)), x, 'bo')
plt.show()