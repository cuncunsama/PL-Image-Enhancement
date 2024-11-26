import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)

def frn(x, slope):
    return 0.4591*leaky_relu(0.5267*leaky_relu(x, slope)+0.4733*gelu(x), slope) + 0.5497*gelu(x)

# class FRN(nn.Module):
#     def __init__(self, relu_slope, inplace):
#         super(FRN, self).__init__()
#         self.relu1 = nn.LeakyReLU(relu_slope, inplace=inplace)
#         self.relu2 = nn.LeakyReLU(relu_slope, inplace=inplace)
#         self.gelu1 = nn.GELU()
#         self.gelu2 = nn.GELU()
        
#     def forward(self, x):
#         out1 = 0.5267*self.relu1(x)
#         out2 = 0.4733*self.gelu1(x)
#         out3 = 0.5497*self.gelu2(x)
#         return 0.4591*self.relu2(out1+out2) + out3
# 定义x的范围
x = np.linspace(-6, 6, 10000)
y = frn(x, 0.2)

# 绘制函数曲线
plt.plot(x, y, label='FRN')
plt.title('FRN Function')
plt.xlabel('x')
plt.ylabel('FRN(x)')
plt.legend()
plt.grid(True)
plt.savefig('example_plot.svg', format='svg', dpi=1000)
plt.show()
