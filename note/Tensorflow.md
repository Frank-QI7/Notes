tf.placeholder(dtype, shape=None, name=None)

此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值

```python
#Create a variable.（创建一个变量）
w = tf.Variable(<initial-value>, name=<optional-name>)
```

tf.matmul() 就是点积

```python
# 2-D tensor `a`
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                      [4. 5. 6.]]
# 2-D tensor `b`
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                         [9. 10.]
                                                         [11. 12.]]
c = tf.matmul(a, b) => [[58 64]
                        [139 154]]

# 3-D tensor `a`
a = tf.constant(np.arange(1, 13, dtype=np.int32),
                shape=[2, 2, 3])                  => [[[ 1.  2.  3.]
                                                       [ 4.  5.  6.]],
                                                      [[ 7.  8.  9.]
                                                       [10. 11. 12.]]]

# 3-D tensor `b`
b = tf.constant(np.arange(13, 25, dtype=np.int32),
                shape=[2, 3, 2])                   => [[[13. 14.]
                                                        [15. 16.]
                                                        [17. 18.]],
                                                       [[19. 20.]
                                                        [21. 22.]
                                                        [23. 24.]]]
c = tf.matmul(a, b) => [[[ 94 100]
                         [229 244]],
                        [[508 532]
                         [697 730]]]
```

![image-20181006170157719](/Users/yunqingqi/Desktop/note/image-20181006170157719.png)

![image-20181006170215167](/Users/yunqingqi/Desktop/note/image-20181006170215167.png)

![image-20181006170333153](/Users/yunqingqi/Desktop/note/image-20181006170333153.png)

![image-20181006170426393](/Users/yunqingqi/Desktop/note/image-20181006170426393.png)

![image-20181006170529449](/Users/yunqingqi/Desktop/note/image-20181006170529449.png)

**进入session后，我们就要给之前定义好的 x,w1,w2,y 一些 concrete data that will be fed to the graph**

![image-20181006170734560](/Users/yunqingqi/Desktop/note/image-20181006170734560.png)

**至此，我们完成了一次前向和反向传播 and it only takes a couple extra lines if we actually want to train the network**

![image-20181006171253756](/Users/yunqingqi/Desktop/note/image-20181006171253756.png)

**但是这样会产生一个问题，那就是每次更新权重的时候，w1 和 w2 是用numpy存储的，是cpu的工作，gpu每次都要从cpu那里获取 w1 和 w2, 速度不匹配，导致训练十分慢，下面是解决办法：**

![image-20181006172426809](/Users/yunqingqi/Desktop/note/image-20181006172426809.png)

**将 w2 和 w2 用变量定义，另外一个不同就是：**

![image-20181006172627109](/Users/yunqingqi/Desktop/note/image-20181006172627109.png)

**我们把 w1 和 w2 的更新方式也写在graph部分，这样的话，权重就一直在GPU里了**

![image-20181006172848716](/Users/yunqingqi/Desktop/note/image-20181006172848716.png)

<font color=#B22222 >**到这里其实还不行，如果运行代码，并输出loss的图的话，我们会发现loss根本没有 go down，而且我们的代码只是计算了loss，并没有对权重进行更新，说明我们的assign calls not catually being exucuted ，so we need to explicitly tell TensorFlow to perform those update operations**</font>

![image-20181006174629118](/Users/yunqingqi/Desktop/note/image-20181006174629118.png)

<font color=#B22222 >**最终版代码就是下图：Tensorflow 还实现了一个optimizer函数，让这些过程变得简单！！！！！**</font>

![image-20181006174721204](/Users/yunqingqi/Desktop/note/image-20181006174721204.png)

**optimizer = tf.train.GradientDescentOptimizier可以把这个换成其他的梯度下降算法，这里到参数就是(learning rate), 然后updates = optimizer.minize(loss) ,这句话就会使每次更新的时候，optimizer 会去graph自动找节点，更新权重 **

**可以看到这样写很麻烦，其实TensorFlow里有一些包可以使用：** 用的时候可以查一查别人怎么用的

![image-20181006182747116](/Users/yunqingqi/Desktop/note/image-20181006182747116.png)

![image-20181006183631839](/Users/yunqingqi/Desktop/note/image-20181006183631839.png)

![image-20181006183744974](/Users/yunqingqi/Desktop/note/image-20181006183744974.png)

![image-20181006183803441](/Users/yunqingqi/Desktop/note/image-20181006183803441.png)

![image-20181006183837094](/Users/yunqingqi/Desktop/note/image-20181006183837094.png)

![image-20181006184722338](/Users/yunqingqi/Desktop/note/image-20181006184722338.png)

![image-20181006185031831](/Users/yunqingqi/Desktop/note/image-20181006185031831.png)

<font color=#B22222>**static的好处就是，一旦你serialize the graph，you have this data structure in memory that represents the entire structure of your network. And now you could take that data structure and just serialize it to disk. And now you've got the whole structure of your network saved in some file. And then you could later rear(培养，栽种，后面) load that thing and then run that computational graph without access to the original code that built it. You might imagine that you might want to train your in Python because it's maybe easier to work with, but then after you serialize that network and then you could deploy ot now in maybe a c++ environment where you don't need to use the original code to build the graph. So that's kind of a nice advantage of static graphs. **</font>

![image-20181006185956452](/Users/yunqingqi/Desktop/note/image-20181006185956452.png)

![image-20181006191719669](/Users/yunqingqi/Desktop/note/image-20181006191719669.png)

**如果想实现上面这样的结构，得用下面这样的方法**：

![image-20181006191748008](/Users/yunqingqi/Desktop/note/image-20181006191748008.png)

可以看出来，TensorFlow 几乎是把所有的功能用自己的语言重写了，而Pytorch 可以让我们使用python等其他语言。

![image-20181006192143070](/Users/yunqingqi/Desktop/note/image-20181006192143070.png)

![image-20181006192332603](/Users/yunqingqi/Desktop/note/image-20181006192332603.png)

**当我们想用dynamic graph 的时候，还是首选 Pytorch**



# Pytorch 神经网络基础

## 1.1 Pytorch & Numpy

### 1.1.1 用Torch还是Numpy

Torch 自称为神经网络界的 Numpy, 因为他能将 torch 产生的 tensor 放在 GPU 中加速运算 (前提是你有合适的 GPU), 就像 Numpy 会把 array 放在 CPU 中加速运算. 所以神经网络的话, 当然是用 Torch 的 tensor 形式数据最好咯. 就像 Tensorflow 当中的 tensor 一样.

当然, 我们对 Numpy 还是爱不释手的, 因为我们太习惯 numpy 的形式了. 不过 torch 看出来我们的喜爱, 他把 torch 做的和 numpy 能很好的兼容. 比如这样就能**自由地转换 numpy array 和 torch tensor** 了:
 numpy array转换成 torch tensor：`torch.from_numpy(np_data)`
 torch tensor转换成 numpy array：`torch_data.numpy()`

```python
import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)
```

### 1.1.2 Torch 中的数学运算

其实 torch 中 tensor 的运算和 numpy array 的如出一辙, 我们就以对比的形式来看. 如果想了解 torch 中其它更多有用的运算符, [可以参考Pytorch中文手册](https://link.jianshu.com?t=http%3A%2F%2Fpytorch-cn.readthedocs.io%2Fzh%2Flatest%2Fpackage_references%2Ftorch%2F).

- **abs 绝对值计算**

```python
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)
```

- **sin 三角函数**

```python
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)
```

- **mean 均值**

```python
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)
```

除了简单的计算, 矩阵运算才是神经网络中最重要的部分. 所以我们展示下矩阵的乘法. 注意一下包含了一个 numpy 中可行, 但是 torch 中不可行的方式.

- **matrix multiplication 矩阵点乘**

```python
# matrix multiplication 矩阵点乘
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)

# !!!!  下面是错误的方法 !!!!
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] 在numpy 中可行
    '\ntorch: ', tensor.dot(tensor)     # torch.dot只能处理一维数组
)
```

## 1.2 变量 Variable

### 1.2.1 什么是Variable

在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯. 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.

我们定义一个 Variable:

```python
import torch
from torch.autograd import Variable # torch 中 Variable 模块

# 先生鸡蛋
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable)
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
```

### 1.2.2 Variable 计算 梯度

我们再对比一下 tensor 的计算和 variable 的计算.

```python
t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5
```

到目前为止, 我们看不出什么不同, 但是时刻记住, Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图(computational graph). 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力啦.

`v_out = torch.mean(variable*variable)` 就是在计算图中添加的一个计算步骤, 计算误差反向传递的时候有他一份功劳, 我们就来举个例子:

```python
v_out.backward()    # 模拟 v_out 的误差反向传递

# 下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2

print(variable.grad)    # 初始 Variable 的梯度
'''
 0.5000  1.0000
 1.5000  2.0000
'''
```

### 1.2.3 获取 Variable 里面的数据

直接`print(variable)`只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.

```python
print(variable)     #  Variable 形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy 形式
"""
[[ 1.  2.]
 [ 3.  4.]]
"""
```

## 1.3 Torch中的激励函数

Torch 中的激励函数有很多, 不过我们平时要用到的就这几个. relu, sigmoid, tanh, softplus. 那我们就看看他们各自长什么样啦.

```python
import torch
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable

# 做一些假数据来观看图像
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x)
```

接着就是做生成不同的激励函数数据:

```python
x_np = x.data.numpy()   # 换成 numpy array, 出图时用

# 几种常用的 激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)  softmax 比较特殊, 不能直接显示, 不过他是关于概率的, 用于分类
```

 接着我们开始画图, 画图的代码也在下面: 

![image-20181017154038480](/Users/yunqingqi/Desktop/note/image-20181017154038480.png)

教程:

```python
import matplotlib.pyplot as plt 

# (8,6)表示整个图片的大小，感觉这个大小正好，1代表名字
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
```

# 建造第一个神经网络

## 2.1 关系拟合

本节会来见证神经网络是如何通过简单的形式将一群数据用一条线条来表示. 或者说, 是如何在数据当中找到他们的关系, 然后用神经网络模型来建立一个可以代表他们关系的线条.

- [本节的全部代码](https://link.jianshu.com?t=https%3A%2F%2Fgithub.com%2FMorvanZhou%2FPyTorch-Tutorial%2Fblob%2Fmaster%2Ftutorial-contents%2F301_regression.py)

### 2.1.1 建立数据集

我们创建一些假数据来模拟真实的情况. 比如一个一元二次函数: y = a * x^2 + b, 我们给 y 数据加上一点噪声来更加真实的展示它.

[为了更好理解下面的代码，首先来看一下Pytorch中view()、squeeze()、unsqueeze()、torch.max()函数](https://blog.csdn.net/lanse_zhicheng/article/details/79148678)

[unsqueeze()](https://www.jianshu.com/p/2eaee422d444)

```python
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 用 Variable 来修饰这些数据 tensor
x, y = torch.autograd.Variable(x), Variable(y)

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
```

### 2.1.2 建立神经网络

建立一个神经网络我们可以直接运用 torch 中的体系. 先定义所有的层属性(`__init__()`), 然后再一层层搭建(`forward(x)`)层于层的关系链接. 建立关系的时候, 我们会用到激励函数.

```python
import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
```

### 2.1.3 训练网络

训练的步骤很简单, 如下:

```python
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

for t in range(100):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```

### 2.1.4 可视化训练过程

为了可视化整个训练的过程, 更好的理解是如何训练, 我们如下操作:

```python
import matplotlib.pyplot as plt

plt.ion()   # 画图
plt.show()

for t in range(100):

    ...
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()   # 清除原有图像
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
```

### 最终代码如下：

```python
import torch
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 用 Variable 来修饰这些数据 tensor
x, y = torch.autograd.Variable(x), Variable(y)

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
plt.ion()   # 画图

for t in range(100):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.show()
```



## 2.2 区分类型（分类）

这次我们也是用最简单的途径来看看神经网络是怎么进行事物的分类.

- [本节的全部代码](https://link.jianshu.com?t=https%3A%2F%2Fgithub.com%2FMorvanZhou%2FPyTorch-Tutorial%2Fblob%2Fmaster%2Ftutorial-contents%2F302_classification.py)

### 2.2.1 建立数据集

我们创建一些假数据来模拟真实的情况. 比如两个二项分布的数据, 不过他们的均值都不一样.

```python
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 假数据  就是制造了 x0，label为y0 ， x1，label为y1的数据
n_data = torch.ones(100, 2)         # 数据的基本形态
# torch.normal 正态分布
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# torch 只能在 Variable 上训练, 所以把它们变成 Variable
x, y = Variable(x), Variable(y)

# 画图
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
```

### 2.2.2 建立神经网络

建立一个神经网络我们可以直接运用 torch 中的体系. 先定义所有的层属性(**init**()), 然后再一层层搭建(forward(x))层于层的关系链接. 这个和我们在前面 regression 的时候的神经网络基本没差. 建立关系的时候, 我们会用到激励函数。

```python
import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2) # 几个类别就几个 output

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (2 -> 10)
  (out): Linear (10 -> 2)
)
"""
```

### 2.2.3 训练网络

训练的步骤很简单, 如下:

```python
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)     # 喂给 net 训练数据 x, 输出分析值

    loss = loss_func(out, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```

### 2.2.4 可视化训练过程

为了可视化整个训练的过程, 更好的理解是如何训练, 我们如下操作:

```python
import matplotlib.pyplot as plt

plt.ion()   # 画图
plt.show()

for t in range(100):

    ...
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
```

## 2.3 快速搭建法

Torch 中提供了很多方便的途径, 同样是神经网络, 能快则快, 我们看看如何用更简单的方式搭建同样的回归神经网络。

- [本节的全部代码](https://link.jianshu.com?t=https%3A%2F%2Fgithub.com%2FMorvanZhou%2FPyTorch-Tutorial%2Fblob%2Fmaster%2Ftutorial-contents%2F303_build_nn_quickly.py)

### 2.3.1 快速搭建

我们先看看之前写神经网络时用到的步骤. 我们用 `net1` 代表这种方式搭建的神经网络.

```python
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net1 = Net(1, 10, 1)   # 这是我们用这种方式搭建的 net1
```

我们用 class 继承了一个 torch 中的神经网络结构, 然后对其进行了修改, 不过还有更快的一招, 用一句话就概括了上面所有的内容!

```python
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
```

我们再对比一下两者的结构:

```python
print(net1)
"""
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
"""
print(net2)
"""
Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=1, bias=True)
)
"""
```

我们会发现 `net2` 多显示了一些内容, 这是为什么呢? 原来他把激励函数也一同纳入进去了, 但是 `net1` 中, 激励函数实际上是在 forward() 功能中才被调用的. 这也就说明了, 相比 `net2`, `net1` 的好处就是, 你可以根据你的个人需要更加个性化你自己的前向传播过程, 比如(RNN). 不过如果你不需要七七八八的过程, 相信 `net2` 这种形式更适合你.

### 再来看一个 forward 函数里用nn.Sequential 来完成前向传播：

```python
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

        ## 最重要的就是这里！！！！！ forward函数的参数一定要有你的输入！！！！！！
        ## 还有就是我们可以在 forward 函数里直接用nn.Sequential()，比如我们之前定义好了
        ## 一个layer1， 我们forward里直接用layer1(x) 并给layer1 赋予我们的  输入 x ！！！
        ## 这样就完成一个网络的前向传播！！！！！！！！！！！！！！！！！！！！！！！
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn = CNN()
```



## 2.4 保存提取

训练好了一个模型, 我们当然想要保存它, 留到下次要用的时候直接提取直接用, 这就是这节的内容啦. 我们用回归的神经网络举例实现保存提取.

- [本节的全部代码](https://link.jianshu.com?t=https%3A%2F%2Fgithub.com%2FMorvanZhou%2FPyTorch-Tutorial%2Fblob%2Fmaster%2Ftutorial-contents%2F304_save_reload.py)

### 2.4.1 保存

我们快速地建造数据, 搭建网络:

```python
import torch
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
```

接下来我们有两种途径来保存:

```python
# 2 ways to save the net
torch.save(net1, 'net.pkl')  # save entire net
torch.save(net1.state_dict(), 'net_params.pkl')   # save only the parameters (速度快, 占内存少)
```

### 2.4.2 提取网络

这种方式将会提取整个神经网络, 网络大的时候可能会比较慢.

```python
def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
```

### 2.4.3 只提取网络参数

这种方式将会提取所有的参数, 然后再放到你的新建网络中.

```python
def restore_params():
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()
```

### 2.4.4 显示结果

调用上面建立的几个功能, 然后出图.

```
# 保存 net1 (1. 整个网络, 2. 只有参数)
save()

# 提取整个网络
restore_net()

# 提取网络参数, 复制到新网络
restore_params()
```

## 2.5 批训练

Torch 中提供了一种帮你整理你的数据结构的好东西, 叫做 DataLoader, 我们能用它来包装自己的数据, 进行批训练.

- [本节的全部代码](https://link.jianshu.com?t=https%3A%2F%2Fgithub.com%2FMorvanZhou%2FPyTorch-Tutorial%2Fblob%2Fmaster%2Ftutorial-contents%2F305_batch_train.py)

### 2.5.1 DataLoader

DataLoader 是 torch 给你用来包装你的数据的工具. 所以你要将自己的 (numpy array 或其他) 数据形式装换成 Tensor, 然后再放进这个包装器中. 使用 DataLoader 有什么好处呢? 就是他们帮你有效地迭代数据, 举例:

```python
import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5      # 批训练的数据个数

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

def show_batch():
    for epoch in range(3):   # train entire dataset 3 times 训练所有!整套!数据 3 次
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step 每一步 loader 释放一小批数据用来学习
            # train your data... 假设这里就是你训练的地方...
            # 打出来一些数据
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())
                  
if __name__ == '__main__':
    show_batch()

"""
Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
"""
```

**可以看出，一次epoch就是把所有数据训练一次**

可以看出, 每步都导出了5个数据进行学习. 然后每个 epoch 的导出数据都是先打乱了以后再导出.
 真正方便的还不是这点. 如果我们改变一下 `BATCH_SIZE = 8`, 这样我们就知道, `step=0` 会导出8个数据, 但是, step=1 时数据库中的数据不够 8个, 这时怎么办呢:

```python
BATCH_SIZE = 8      # 批训练的数据个数

...

for ...:
    for ...:
        ...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
"""
Epoch:  0 | Step:  0 | batch x:  [  6.   7.   2.   3.   1.   9.  10.   4.] | batch y:  [  5.   4.   9.   8.  10.   2.   1.   7.]
Epoch:  0 | Step:  1 | batch x:  [ 8.  5.] | batch y:  [ 3.  6.]
Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.   1.   7.   8.] | batch y:  [  8.   7.   9.   2.   1.  10.   4.   3.]
Epoch:  1 | Step:  1 | batch x:  [ 5.  6.] | batch y:  [ 6.  5.]
Epoch:  2 | Step:  0 | batch x:  [  3.   9.   2.   6.   7.  10.   4.   8.] | batch y:  [ 8.  2.  9.  5.  4.  1.  7.  3.]
Epoch:  2 | Step:  1 | batch x:  [ 1.  5.] | batch y:  [ 10.   6.]
"""
```

这时, 在 step=1 就只给你返回这个 epoch 中剩下的数据就好了.

## 2.6 加速神经网络训练

加速你的神经网络训练过程包括以下几种模式:

- Stochastic Gradient Descent (SGD)
- Momentum
- AdaGrad
- RMSProp
- Adam

具体算法讲解待后续补充，也可网上搜索相关资料。

## 2.7 Optimizer 优化器

 这节内容主要是用 Torch 实践上一小节中提到的几种优化器, 这节内容的最后会对比各种优化器的效果

- [本节的全部代码](https://link.jianshu.com?t=https%3A%2F%2Fgithub.com%2FMorvanZhou%2FPyTorch-Tutorial%2Fblob%2Fmaster%2Ftutorial-contents%2F306_optimizer.py)

### 2.7.1 伪数据

 为了对比各种优化器的效果, 我们需要有一些数据, 今天我们还是自己编一些伪数据, 这批数据是这样的:

![image-20181017172149427](/Users/yunqingqi/Desktop/note/image-20181017172149427.png)

### 2.7.2 每个优化器优化一个神经网络

为了对比每一种优化器, 我们给他们各自创建一个神经网络, 但这个神经网络都来自同一个 Net 形式.

```python
# 默认的 network 形式
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# 为每个优化器创建一个 net
net_SGD         = Net()
net_Momentum    = Net()
net_RMSprop     = Net()
net_Adam        = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
```

### 2.7.3 优化器 Optimizer

接下来在创建不同的优化器, 用来训练不同的网络. 并创建一个 `loss_func` 用来计算误差. 我们用几种常见的优化器, SGD, Momentum, RMSprop, Adam.

```python
# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss
```

### 2.7.3 优化器 Optimizer

接下来在创建不同的优化器, 用来训练不同的网络. 并创建一个 `loss_func` 用来计算误差. 我们用几种常见的优化器, SGD, Momentum, RMSprop, Adam.

```python
# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss
```

### 2.7.4 训练、出图

接下来训练和 loss 画图.

```python
# 训练
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)  # 务必要用 Variable 包一下
        b_y = Variable(batch_y)

        # 对每个优化器, 优化属于他的神经网络
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)              # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            l_his.append(loss.data[0])     # loss recoder

# 出图
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
```

![image-20181017172128903](/Users/yunqingqi/Desktop/note/image-20181017172128903.png)

SGD 是最普通的优化器, 也可以说没有加速效果, 而 Momentum 是 SGD 的改良版, 它加入了动量原则. 后面的 RMSprop 又是 Momentum 的升级版. 而 Adam 又是 RMSprop 的升级版. 不过从这个结果中我们看到, Adam 的效果似乎比 RMSprop 要差一点. 所以说并不是越先进的优化器, 结果越佳. 我们在自己的试验中可以尝试不同的优化器, 找到那个最适合你数据/网络的优化器.

# 高级神经网络结构

剩下的内容就直接去

PycharmProject里的 PyTorch_tutorial下面，打开jupyter notebook 看吧

**Pytorch.nn.conv2d 的使用：**

![image-20181017201423805](/Users/yunqingqi/Desktop/note/image-20181017201423805.png)、

#### Pytorch 中的 contiguous

contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。 
一种可能的解释是： 
有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。 
判断是否contiguous用torch.Tensor.is_contiguous()函数。

```python
import torch
x = torch.ones(10, 10)
x.is_contiguous()  # True
x.transpose(0, 1).is_contiguous()  # False
x.transpose(0, 1).contiguous().is_contiguous()  # True
```

在pytorch的最新版本0.4版本中，增加了torch.reshape(), 这与 numpy.reshape 的功能类似。它大致相当于 tensor.contiguous().view()

#### 还有一个就是 transpose的使用

```python
transpose((2,0,1)) 参数里面加括号，就是按照固定括号里的顺序来
另外一个是：
transpose(1,2)  就是把第 1 维 和 第 2 维 交换！！！！！
```

x_y_offset = torch.cat((x_offset, y_offset), 1)  把 第 2 维度相加，之前x_offset, y_offset 的 第二维度都是1，现在x_y_offset的维度就是（169，2）

下面是numpy.meshgrid的作用：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
y = np.array([0, 1])

X, Y = np.meshgrid(x, y)
print(X)
print(Y)

# 从输出的结果来看，两种方法生成的坐标矩阵一毛一样。
[[0 1 2]
 [0 1 2]]
[[0 0 0]
 [1 1 1]]

！～！～！！～一个是列方向重复，一个是把每一个行元素重复

```

Torch.repeat() 函数

```python
>>> x = torch.Tensor([1, 2, 3])
>>> x.repeat(4, 2)
 1  2  3  1  2  3
 1  2  3  1  2  3
 1  2  3  1  2  3
 1  2  3  1  2  3
[torch.FloatTensor of size 4x6]
>>> x.repeat(4, 2, 1).size()
torch.Size([4, 2, 3])

还有一个例子就是 在做yolo3的时候，有 
x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,3)
最后 x_y_offset.shape 就是 ([169, 6])
```

torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数

下面这个需要注意：比如一个数组的维度是（2，3，5），我们只访问最后一维的一个元素时，那么输出的结果的维度就是（2，3）

```python
np.set_printoptions(threshold=np.nan)
test = np.random.randint(10, size=(2, 3, 5))
print("test: ",test)
print("test[:, :, 4]: ",test[:, :, 4].shape)
```

下面这个是unsqueeze的另外一个用法：（原来test的[][][4]的值如果小于某一个阈值，我们就让这个元素的整个[][][]为0 ！！！！）

```python
test = np.random.randint(10, size=(2, 3, 5))
test = torch.from_numpy(test)

print(test[:, :, 4] > 5,(test[:, :, 4] > 5).shape)
conf_mask = (test[:, :, 4] > 5).float().unsqueeze(2)
print(conf_mask,conf_mask.shape)
# 结果是这样的 ##########
tensor([[0, 0, 1],
        [0, 1, 0]], dtype=torch.uint8) torch.Size([2, 3])
tensor([[[0.],
         [0.],
         [1.]],

        [[0.],
         [1.],
         [0.]]]) torch.Size([2, 3, 1])
这样做的目的是： 原来test的[][][4]的值如果小于某一个阈值，我们就让这个元素的整个[][][]为0 ！！！！
```

touch.new的用法就是 构建一个有相同数据类型的tensor

```python
box_corner = prediction.new(prediction.shape)
```

