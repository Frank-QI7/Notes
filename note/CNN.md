![image-20181022175750072](/Users/yunqingqi/Desktop/note/image-20181022175750072.png)

![image-20180908160326871](/Users/yunqingqi/Desktop/note/image-20180908160326871.png)

其实，有时候我们可以简化我们的computational graph like this：

![image-20180908160921454](/Users/yunqingqi/Desktop/note/image-20180908160921454.png)

一个有趣的现象是在多数情况下，反向传播中的梯度可以被很直观地解释。例如神经网络中最常用的加法、乘法和取最大值这三个门单元，它们在反向传播过程中的行为都有非常简单的解释:

![image-20180908161526767](/Users/yunqingqi/Desktop/note/image-20180908161526767.png)

**加法门单元**把输出的梯度相等地分发给它所有的输入，这一行为与输入值在前向传播时的值无关。这是因为加法操作的局部梯度都是简单的+1，所以所有输入的梯度实际上就等于输出的梯度，因为乘以1.0保持不变。上例中，加法门把梯度2.00不变且相等地路由给了两个输入。

**取最大值门单元**对梯度做路由。和加法门不同，取最大值门将梯度转给其中一个输入，这个输入是在前向传播中值最大的那个输入。这是因为在取最大值门中，最高值的局部梯度是1.0，其余的是0。上例中，取最大值门将梯度2.00转给了**z**变量，因为**z**的值比**w**高，于是**w**的梯度保持为0。

**乘法门单元**相对不容易解释。它的局部梯度就是输入值，但是是相互交换之后的，然后根据链式法则乘以输出值的梯度。上例中，**x**的梯度是-4.00x2.00=-8.00。

加法操作将梯度相等地分发给它的输入。取最大操作将梯度路由给更大的输入。乘法门拿取输入激活数据，对它们进行交换，然后乘以梯度。

*非直观影响及其结果*。注意一种比较特殊的情况，如果乘法门单元的其中一个输入非常小，而另一个输入非常大，那么乘法门的操作将会不是那么直观：**它将会把大的梯度分配给小的输入，把小的梯度分配给大的输入。在线性分类器中，权重和输入是进行点积![image-20180908162845043](/Users/yunqingqi/Desktop/note/image-20180908162845043.png) 这说明输入数据的大小对于权重梯度的大小有影响。例如，在计算过程中对所有输入数据样本xi乘以1000，那么权重的梯度将会增大1000倍，这样就必须降低学习率来弥补**。这就是为什么数据预处理关系重大，它即使只是有微小变化，也会产生巨大影响。对于梯度在计算线路中是如何流动的有一个直观的理解，可以帮助读者调试网络。

```python
w = [2,-3,-3] # 假设一些随机数据和权重
x = [-1, -2]

# 前向传播
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid函数

# 对神经元反向传播
ddot = (1 - f) * f # 点积变量的梯度, 使用sigmoid函数求导
dx = [w[0] * ddot, w[1] * ddot] # 回传到x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # 回传到w
# 完成！得到输入的梯度
```

![image-20180908162954705](/Users/yunqingqi/Desktop/note/image-20180908162954705.png)

![image-20180908164932429](/Users/yunqingqi/Desktop/note/image-20180908164932429.png)

![image-20180908164949490](/Users/yunqingqi/Desktop/note/image-20180908164949490.png)

![image-20180908165013769](/Users/yunqingqi/Desktop/note/image-20180908165013769.png)

![image-20180908165814032](/Users/yunqingqi/Desktop/note/image-20180908165814032.png)

下面来理解一下矩阵的导数：

![image-20180908171921245](/Users/yunqingqi/Desktop/note/image-20180908171921245.png)

![image-20180908171938040](/Users/yunqingqi/Desktop/note/image-20180908171938040.png)

这就和之前看的jacobian一样了

这也就可以理解下面的 xT 了

![image-20180908172524168](/Users/yunqingqi/Desktop/note/image-20180908172524168.png)

![image-20180908173857852](/Users/yunqingqi/Desktop/note/image-20180908173857852.png)

注意，shape要使矩阵可以相乘

![image-20180908182332076](/Users/yunqingqi/Desktop/note/image-20180908182332076.png)

[Backpropagation](https://zh.wikipedia.org/wiki/反向传播算法)  直接看推导

![image-20181002182825275](/Users/yunqingqi/Desktop/note/image-20181002182825275.png)

<font color=#B22222 >这一步是最重要的！！！！ </font>

**下面来看一个例题，来更好的理解**

![image-20181002183014703](/Users/yunqingqi/Desktop/note/image-20181002183014703.png)

![image-20181002183038844](/Users/yunqingqi/Desktop/note/image-20181002183038844.png)

<font color=#B22222 > 在计算Error for hidden layers的时候，要注意看上面wiki里面给的这个公式 </font>

![image-20181002183257467](/Users/yunqingqi/Desktop/note/image-20181002183257467.png)

If j is an inner neuron，我们注意看，怎么求的 ![image-20181002183331612](/Users/yunqingqi/Desktop/note/image-20181002183331612.png)

![image-20181002183348575](/Users/yunqingqi/Desktop/note/image-20181002183348575.png)

$\delta$1 = 从右往左转back时刚计算的$\delta$  乘  back时刚计算的w，这时的o就是自己的output，为0.68

其实就是和backpropagation一样的，只是说，我们在处理激活函数的时候，多了个(1-o)o,看之前那个画的图的纸

**再看一个带learning rate的例子**

![image-20181002184615149](/Users/yunqingqi/Desktop/note/image-20181002184615149.png)

![image-20181002184631671](/Users/yunqingqi/Desktop/note/image-20181002184631671.png)

**需要记住的是：不应该因为害怕出现过拟合而使用小网络。相反，应该进尽可能使用大网络，然后使用正则化技巧来控制过拟合。**

![image-20181002204013857](/Users/yunqingqi/Desktop/note/image-20181002204013857.png)

一般来说我们会把32x32x3的图片拉长成 3072x1 的矩阵，**但是为了保存图片的空间特性，我们用 卷积层 **

卷积层就是说，我们的weight是一个5x5x3的filter，slide over the image spatially.

**Filters always extend the full depth of the input volume**，和输入一样，最后以层都是3，所以它们都是一个很小的spatial area

![image-20181002204343140](/Users/yunqingqi/Desktop/note/image-20181002204343140.png)

当我们做 dot product 的时候，我们会把这个filter拉成一维的vector来做计算

![image-20181002205030633](/Users/yunqingqi/Desktop/note/image-20181002205030633.png)

这个filter会 slide over all spatial locations，所以最后输出的是28x28x1的矩阵

![image-20181002210100133](/Users/yunqingqi/Desktop/note/image-20181002210100133.png)

然后我们还可以添加第二个filter

![image-20181002210223565](/Users/yunqingqi/Desktop/note/image-20181002210223565.png)

**ConvNet is a sequence of Convolution Layers, interspersed with activation functions**

![image-20181002210543802](/Users/yunqingqi/Desktop/note/image-20181002210543802.png)

![image-20181002210908530](/Users/yunqingqi/Desktop/note/image-20181002210908530.png)

前面几层的卷积核一般代表了low level features，比如边缘特征(edges), middle level 的卷积核会得到一些更加复杂的图像特征，比如边角，斑点。

![image-20181002212302777](/Users/yunqingqi/Desktop/note/image-20181002212302777.png)

![image-20181002213155272](/Users/yunqingqi/Desktop/note/image-20181002213155272.png)

当我们用3x3的filter的时候，一般会用0补一个pad。。。我们用 zero pad 是为了matain input size 和一些重要的信息

![image-20181002213614631](/Users/yunqingqi/Desktop/note/image-20181002213614631.png)

![image-20181002213814942](/Users/yunqingqi/Desktop/note/image-20181002213814942.png)

![image-20181002214018377](/Users/yunqingqi/Desktop/note/image-20181002214018377.png)

![image-20181002214252150](/Users/yunqingqi/Desktop/note/image-20181002214252150.png)

 padding的用途：保持边界信息，如果不加padding层的话，最边缘的像素点信息只会卷积核被扫描到一次，但是图像中间的像素点会被扫描到很多遍，那么就会在一定程度上降低边界信息的参考程度，但是在加了padding之后，在实际处理过程中就会从新的边界进行扫描，就从一定程度上解决了这些问题。还有一点就是可以利用padding层来对输入尺寸有差异图片进行补齐，是的输入图片尺寸一致。

​    池化层的功能：第一，又进行了一次特征提取，所以能减小下一层数据的处理量。

​                            第二，能够获得更为抽象的信息，从而防止过拟合，也就是提高了一定的泛化性

​                            第三，由于这种抽象性，所以能对输入的微小变化产生更大的容忍，也就是保持了它的不变性，这里的容忍包括图像的少量平移、旋转缩放等操作变化。

*常规神经网络对于大尺寸图像效果不尽人意*。在CIFAR-10中，图像的尺寸是32x32x3（宽高均为32像素，3个颜色通道），因此，对应的的常规神经网络的第一个隐层中，每一个单独的全连接神经元就有32x32x3=3072个权重。这个数量看起来还可以接受，但是很显然这个全连接的结构不适用于更大尺寸的图像。举例说来，一个尺寸为200x200x3的图像，会让神经元包含200x200x3=120,000个权重值。而网络中肯定不止一个神经元，那么参数的量就会快速增加！显而易见，这种全连接方式效率低下，大量的参数也很快会导致网络过拟合。

**参数共享**：作一个合理的假设：如果一个特征在计算某个空间位置(x,y)的时候有用，那么它在计算另一个不同位置(x2,y2)的时候也有用。基于这个假设，可以显著地减少参数数量。换言之，就是将深度维度上一个单独的2维切片看做**深度切片（depth slice）**，比如一个数据体尺寸为[55x55x96]的就有96个深度切片，每个尺寸为[55x55]。在每个深度切片上的神经元都使用同样的权重和偏差。在这样的参数共享下，例子中的第一个卷积层就只有96个不同的权重集了，一个权重集对应一个深度切片，共有96x11x11x3=34,848个不同的权重，或34,944个参数（+96个偏差）。在每个深度切片中的55x55个权重使用的都是同样的参数。在反向传播的时候，都要计算每个神经元对它的权重的梯度，但是需要把同一个深度切片上的所有神经元对权重的梯度累加，这样就得到了对共享权重的梯度。这样，每个切片只更新一个权重集。

注意，如果在一个深度切片中的所有权重都使用同一个权重向量，那么卷积层的前向传播在每个深度切片中可以看做是在计算神经元权重和输入数据体的**卷积**（这就是“卷积层”名字由来）。这也是为什么总是将这些权重集合称为**滤波器（filter）**（或**卷积核（kernel）**），因为它们和输入进行了卷积。

<font color=#B22222 >对于滤波器，也有一定的规则要求：</font>

​      1）滤波器的大小应该是奇数，这样它才有一个中心，例如3x3，5x5或者7x7。有中心了，也有了半径的称呼，例如5x5大小的核的半径就是2。

​      2）滤波器矩阵所有的元素之和应该要等于1，这是为了保证滤波前后图像的亮度保持不变。当然了，这不是硬性要求了。

​      3）如果滤波器矩阵所有元素之和大于1，那么滤波后的图像就会比原图像更亮，反之，如果小于1，那么得到的图像就会变暗。如果和为0，图像不会变黑，但也会非常暗。

如果我们用的filter是3x3大小的，那么卷积操作允许我们只用 9 个参数来获得新的图像，每个输出特性不用**「查看」**每个输入特征，而是只是**「查看」**来自大致相同位置的输入特征。**请注意这一点，因为这对我们后面的讨论至关重要。**

**卷积在计算前需要翻转卷积核，将核围绕中心旋转180度**！！

**图像的求导**：就是相邻pixel的差值！！！！！

```
一般就是指亮度的变化率（梯度）
假设某灰度图像有如下像素区域。
a b c
d e f
g h i 
那么对于中心像素e：
dx＝f-d（或者f-e）
dy＝h-i（或者h-e）
当然如果用sobel算子的话
dx＝（c+2f+i）-（a+2d+g）
dy＝（g+2h+i）-（a+2b+c）
```

![image-20181003143842420](/Users/yunqingqi/Desktop/note/image-20181003143842420.png)

注意有时候参数共享假设可能没有意义，特别是当卷积神经网络的输入图像是一些明确的中心结构时候。这时候我们就应该期望在图片的不同位置学习到完全不同的特征。一个具体的例子就是输入图像是人脸，人脸一般都处于图片中心。你可能期望不同的特征，比如眼睛特征或者头发特征可能（也应该）会在图片的不同位置被学习。在这个例子中，通常就放松参数共享的限制，将层称为**局部连接层**（Locally-Connected Layer）。

**Numpy例子**：为了让讨论更加的具体，我们用代码来展示上述思路。假设输入数据体是numpy数组**X**。那么：

- 一个位于**(x,y)**的深度列（或纤维）将会是**X[x,y,:]**。
- 在深度为**d**处的深度切片，或激活图应该是**X[:,:,d]**。

*卷积层例子*：假设输入数据体**X**的尺寸**X.shape:(11,11,4)**，不使用零填充P=0，滤波器的尺寸是F=5, 步长S=2。那么输出数据体的空间尺寸就是(11-5)/2+1=4，即输出数据体的宽度和高度都是4。那么在输出数据体中的激活映射（称其为**V**）看起来就是下面这样（在这个例子中，只有部分元素被计算）：

- **V[0,0,0] = np.sum(X[:5,:5,:] \* W0) + b0**
- **V[1,0,0] = np.sum(X[2:7,:5,:] \* W0) + b0**
- **V[2,0,0] = np.sum(X[4:9,:5,:] \* W0) + b0**
- **V[3,0,0] = np.sum(X[6:11,:5,:] \* W0) + b0**

在numpy中，*****操作是进行数组间的逐元素相乘。权重向量**W0**是该神经元的权重，**b0**是其偏差。在这里，**W0**被假设尺寸是**W0.shape: (5,5,4)**，因为滤波器的宽高是5，输入数据量的深度是4。注意在每一个点，计算点积的方式和之前的常规神经网络是一样的。同时，计算内积的时候使用的是同一个权重和偏差（因为参数共享），在宽度方向的数字每次上升2（因为步长为2）。要构建输出数据体中的第二张激活图，代码应该是：

- **V[0,0,1] = np.sum(X[:5,:5,:] \* W1) + b1**
- **V[1,0,1] = np.sum(X[2:7,:5,:] \* W1) + b1**
- **V[2,0,1] = np.sum(X[4:9,:5,:] \* W1) + b1**
- **V[3,0,1] = np.sum(X[6:11,:5,:] \* W1) + b1**
- **V[0,1,1] = np.sum(X[:5,2:7,:] \* W1) + b1** （在y方向上）
- **V[2,3,1] = np.sum(X[4:9,6:11,:] \* W1) + b1** （或两个方向上同时）

还有，要记得这些卷积操作通常后面接的是ReLU层

自己之前理解的怎么计算神经网络的想法是错误的，这里有一个正确计算的例子：

<font color=#B22222 >正确理解卷积核和输入的正向传播计算    [斯坦福课程官网例子](http://cs231n.github.io/convolutional-networks/)    </font>

**用矩阵乘法实现**：卷积运算本质上就是在滤波器和输入数据的局部区域间做点积。卷积层的常用实现方式就是利用这一点，将卷积层的前向传播变成一个巨大的矩阵乘法：

1. 输入图像的局部区域被**im2col**操作拉伸为列。比如，如果输入是[227x227x3]，要与尺寸为11x11x3的滤波器以步长为4进行卷积，就取输入中的[11x11x3]数据块，然后将其拉伸为长度为11x11x3=363的列向量。重复进行这一过程，因为步长为4，所以输出的宽高为(227-11)/4+1=55，所以得到*im2col*操作的输出矩阵**X_col**的尺寸是[363x3025]，其中每列是拉伸的感受野，共有55x55=3,025个。注意因为感受野之间有重叠，所以输入数据体中的数字在不同的列中可能有重复。
2. 卷积层的权重也同样被拉伸成行。举例，如果有96个尺寸为[11x11x3]的滤波器，就生成一个矩阵**W_row**，尺寸为[96x363]。
3. 现在卷积的结果和进行一个大矩阵乘**np.dot(W_row, X_col)**是等价的了，能得到每个滤波器和每个感受野间的点积。在我们的例子中，这个操作的输出是[96x3025]，给出了每个滤波器在每个位置的点积输出。
4. 结果最后必须被重新变为合理的输出尺寸[55x55x96]。

这个方法的缺点就是占用内存太多，因为在输入数据体中的某些值在**X_col**中被复制了多次。但是，其优点是矩阵乘法有非常多的高效实现方式，我们都可以使用（比如常用的[BLAS](https://link.zhihu.com/?target=http%3A//www.netlib.org/blas/) API）。还有，同样的*im2col*思路可以用在汇聚操作中。

## 汇聚层

通常，在连续的卷积层之间会周期性地插入一个汇聚层。它的作用是逐渐降低数据体的空间尺寸，这样的话就能减少网络中参数的数量，使得计算资源耗费变少，也能有效控制过拟合。汇聚层使用MAX操作，对输入数据体的每一个深度切片独立进行操作，改变它的空间尺寸。最常见的形式是汇聚层使用尺寸2x2的滤波器，以步长为2来对每个深度切片进行降采样，将其中75%的激活信息都丢掉。每个MAX操作是从4个数字中取最大值（也就是在深度切片中某个2x2的区域）。深度保持不变。汇聚层的一些公式：

- 输入数据体尺寸![W_1\cdot H_1\cdot D_1](https://www.zhihu.com/equation?tex=W_1%5Ccdot+H_1%5Ccdot+D_1)

- 有两个超参数：

- - 空间大小![F](https://www.zhihu.com/equation?tex=F)
  - 步长![S](https://www.zhihu.com/equation?tex=S)

- 输出数据体尺寸![W_2\cdot H_2\cdot D_2](https://www.zhihu.com/equation?tex=W_2%5Ccdot+H_2%5Ccdot+D_2)，其中

![ W_2=(W_1-F)/S+1](https://www.zhihu.com/equation?tex=+W_2%3D%28W_1-F%29%2FS%2B1)

![H_2=(H_1-F)/S+1](https://www.zhihu.com/equation?tex=H_2%3D%28H_1-F%29%2FS%2B1)

![D_2=D_1](https://www.zhihu.com/equation?tex=D_2%3DD_1)

- 因为对输入进行的是固定函数计算，所以没有引入参数
- 在汇聚层中很少使用零填充

在实践中，最大汇聚层通常只有两种形式：一种是![F=3,S=2](https://www.zhihu.com/equation?tex=F%3D3%2CS%3D2)，也叫重叠汇聚（overlapping pooling），另一个更常用的是![F=2,S=2](https://www.zhihu.com/equation?tex=F%3D2%2CS%3D2)。对更大感受野进行汇聚需要的汇聚尺寸也更大，而且往往对网络有破坏性。

![image-20181003174219651](/Users/yunqingqi/Desktop/note/image-20181003174219651.png)

**反向传播：**回顾一下反向传播的内容，其中![max(/Users/yunqingqi/Desktop/note/equation-20181003174510917)](https://www.zhihu.com/equation?tex=max%28x%2Cy%29)函数的反向传播可以简单理解为将梯度只沿最大的数回传。因此，在向前传播经过汇聚层的时候，通常会把池中最大元素的索引记录下来（有时这个也叫作**道岔（switches）**），这样在反向传播的时候梯度的路由就很高效。

**不使用汇聚层**：很多人不喜欢汇聚操作，认为可以不使用它。比如在[Striving for Simplicity: The All Convolutional Net](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1412.6806)一文中，提出使用一种只有重复的卷积层组成的结构，抛弃汇聚层。通过在卷积层中使用更大的步长来降低数据体的尺寸。有发现认为，在训练一个良好的生成模型时，弃用汇聚层也是很重要的。比如变化自编码器（VAEs：variational autoencoders）和生成性对抗网络（GANs：generative adversarial networks）。现在看起来，未来的卷积网络结构中，可能会很少使用甚至不使用汇聚层。

**[Transformation invariance 卷积神经网络的平移不变性](https://zhangting2020.github.io/2018/05/30/Transform-Invariance/)**

#### 层的尺寸设置规律

---

到现在为止，我们都没有提及卷积神经网络中每层的超参数的使用。现在先介绍设置结构尺寸的一般性规则，然后根据这些规则进行讨论：

**输入层**（包含图像的）应该能被2整除很多次。常用数字包括32（比如CIFAR-10），64，96（比如STL-10）或224（比如ImageNet卷积神经网络），384和512。

**卷积层**应该使用小尺寸滤波器（比如3x3或最多5x5），使用步长![S=1](https://www.zhihu.com/equation?tex=S%3D1)。还有一点非常重要，就是对输入数据进行零填充，这样卷积层就不会改变输入数据在空间维度上的尺寸。比如，当![F=3](https://www.zhihu.com/equation?tex=F%3D3)，那就使用![P=1](https://www.zhihu.com/equation?tex=P%3D1)来保持输入尺寸。当![F=5,P=2](https://www.zhihu.com/equation?tex=F%3D5%2CP%3D2)，一般对于任意![F](https://www.zhihu.com/equation?tex=F)，当![P=(F-1)/2](https://www.zhihu.com/equation?tex=P%3D%28F-1%29%2F2)的时候能保持输入尺寸。如果必须使用更大的滤波器尺寸（比如7x7之类），通常只用在第一个面对原始图像的卷积层上。

**汇聚层**负责对输入数据的空间维度进行降采样。最常用的设置是用用2x2感受野（即![F=2](https://www.zhihu.com/equation?tex=F%3D2)）的最大值汇聚，步长为2（![S=2](https://www.zhihu.com/equation?tex=S%3D2)）。注意这一操作将会把输入数据中75%的激活数据丢弃（因为对宽度和高度都进行了2的降采样）。另一个不那么常用的设置是使用3x3的感受野，步长为2。最大值汇聚的感受野尺寸很少有超过3的，因为汇聚操作过于激烈，易造成数据信息丢失，这通常会导致算法性能变差。

*减少尺寸设置的问题*：上文中展示的两种设置是很好的，因为所有的卷积层都能保持其输入数据的空间尺寸，汇聚层只负责对数据体从空间维度进行降采样。如果使用的步长大于1并且不对卷积层的输入数据使用零填充，那么就必须非常仔细地监督输入数据体通过整个卷积神经网络结构的过程，确认所有的步长和滤波器都尺寸互相吻合，卷积神经网络的结构美妙对称地联系在一起。

*为什么在卷积层使用1的步长*？在实际应用中，更小的步长效果更好。上文也已经提过，步长为1可以让空间维度的降采样全部由汇聚层负责，卷积层只负责对输入数据体的深度进行变换。

*为何使用零填充*？使用零填充除了前面提到的可以让卷积层的输出数据保持和输入数据在空间维度的不变，还可以提高算法性能。如果卷积层值进行卷积而不进行零填充，那么数据体的尺寸就会略微减小，那么图像边缘的信息就会过快地损失掉。

*因为内存限制所做的妥协*：在某些案例（尤其是早期的卷积神经网络结构）中，基于前面的各种规则，内存的使用量迅速飙升。例如，使用64个尺寸为3x3的滤波器对224x224x3的图像进行卷积，零填充为1，得到的激活数据体尺寸是[224x224x64]。这个数量就是一千万的激活数据，或者就是72MB的内存（每张图就是这么多，激活函数和梯度都是）。因为GPU通常因为内存导致性能瓶颈，所以做出一些妥协是必须的。在实践中，人们倾向于在网络的第一个卷积层做出妥协。例如，可以妥协可能是在第一个卷积层使用步长为2，尺寸为7x7的滤波器（比如在ZFnet中）。在AlexNet中，滤波器的尺寸的11x11，步长为4。

![image-20181003181611363](/Users/yunqingqi/Desktop/note/image-20181003181611363.png)

## 数据预处理

关于数据预处理我们有3个常用的符号，数据矩阵**X**，假设其尺寸是**[N x D]**（**N**是数据样本的数量，**D**是数据的维度）。

<font color=#B22222 >均值减法（Mean subtraction)是预处理最常用的形式。</font>它对数据中每个独立*特征*减去平均值，从几何上可以理解为在每个维度上都将数据云的中心都迁移到原点。在numpy中，该操作可以通过代码**X -= np.mean(X, axis=0)**实现。而对于图像，更常用的是对所有像素都减去一个值，可以用**X -= np.mean(X)**实现，也可以在3个颜色通道上分别操作。

**归一化（Normalization）**是指将数据的所有维度都归一化，使其数值范围都近似相等。有两种常用方法可以实现归一化。第一种是先对数据做零中心化（zero-centered）处理，然后每个维度都除以其标准差，实现代码为**X /= np.std(X, axis=0)**。第二种方法是对每个维度都做归一化，使得每个维度的最大和最小值是1和-1。这个预处理操作只有在确信不同的输入特征有不同的数值范围（或计量单位）时才有意义，但要注意预处理操作的重要性几乎等同于学习算法本身。在图像处理中，由于像素的数值范围几乎是一致的（都在0-255之间），所以进行这个额外的预处理步骤并不是很必要。

**PCA和白化（Whitening）**是另一种预处理形式。在这种处理中，先对数据进行零中心化处理，然后计算协方差矩阵，它展示了数据中的相关性结构。

```python
# 假设输入数据矩阵X的尺寸为[N x D]
X -= np.mean(X, axis = 0) # 对数据进行零中心化(重要)
cov = np.dot(X.T, X) / X.shape[0] # 得到数据的协方差矩阵
```

数据协方差矩阵的第(i, j)个元素是数据第i个和第j个维度的*协方差*。具体来说，该矩阵的对角线上的元素是方差。还有，协方差矩阵是对称和[半正定](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Positive-definite_matrix%23Negative-definite.2C_semidefinite_and_indefinite_matrices)的。我们可以对数据协方差矩阵进行SVD（奇异值分解）运算。

```python
U,S,V = np.linalg.svd(cov)
```

U的列是特征向量，S是装有奇异值的1维数组（因为cov是对称且半正定的，所以S中元素是特征值的平方）。为了去除数据相关性，将已经零中心化处理过的原始数据投影到特征基准上：

```python
Xrot = np.dot(X,U) # 对数据去相关性
```

注意U的列是标准正交向量的集合（范式为1，列之间标准正交），所以可以把它们看做标准正交基向量。因此，投影对应x中的数据的一个旋转，旋转产生的结果就是新的特征向量。如果计算**Xrot**的协方差矩阵，将会看到它是对角对称的。**np.linalg.svd**的一个良好性质是在它的返回值**U**中，特征向量是按照特征值的大小排列的。我们可以利用这个性质来对数据降维，只要使用前面的小部分特征向量，丢弃掉那些包含的数据没有**方差**的维度。 这个操作也被称为主成分分析（ [Principal Component Analysis](https://link.zhihu.com/?target=http%3A//en.wikipedia.org/wiki/Principal_component_analysis) 简称PCA）降维：

```python
Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced 变成 [N x 100]
```

经过上面的操作，将原始的数据集的大小由[N x D]降到了[N x 100]，留下了数据中包含最大**方差**的100个维度。通常使用PCA降维过的数据训练线性分类器和神经网络会达到非常好的性能效果，同时还能节省时间和存储器空间。

<font color=#B22222 >**常见错误 ：** 进行预处理很重要的一点是：任何预处理策略（比如数据均值）都只能在训练集数据上进行计算，算法训练完毕后再应用到验证集或者测试集上。例如，如果先计算整个数据集图像的平均值然后每张图片都减去平均值，最后将整个数据集分成训练/验证/测试集，那么这个做法是错误的。**应该怎么做呢？应该先分成训练/验证/测试集，只是从训练集中求图片平均值，然后各个集（训练/验证/测试集）中的图像再减去这个平均值。**</font>

**译者注：此处确为初学者常见错误，请务必注意！**

## 权重初始化

**错误：全零初始化。**让我们从应该避免的错误开始。在训练完毕后，虽然不知道网络中每个权重的最终值应该是多少，但如果数据经过了恰当的归一化的话，就可以假设所有权重数值中大约一半为正数，一半为负数。这样，一个听起来蛮合理的想法就是把这些权重的初始值都设为0吧，因为在期望上来说0是最合理的猜测。这个做法错误的！因为如果网络中的每个神经元都计算出同样的输出，然后它们就会在反向传播中计算出同样的梯度，从而进行同样的参数更新。换句话说，如果权重被初始化为同样的值，神经元之间就失去了不对称性的源头

**小随机数初始化。**因此，权重初始值要非常接近0又不能等于0。解决方法就是将权重初始化为很小的数值，以此来*打破对称性*。其思路是：如果神经元刚开始的时候是随机且不相等的，那么它们将计算出不同的更新，并将自身变成整个网络的不同部分。小随机数权重初始化的实现方法是：**W = 0.01 \* np.random.randn(D,H)。**其中**randn**函数是基于零均值和标准差的一个高斯分布（**译者注：国内教程一般习惯称均值参数为期望![\mu](https://www.zhihu.com/equation?tex=%5Cmu)**）来生成随机数的。根据这个式子，每个神经元的权重向量都被初始化为一个随机向量，而这些随机向量又服从一个多变量高斯分布，这样在输入空间中，所有的神经元的指向是随机的。也可以使用均匀分布生成的随机数，但是从实践结果来看，对于算法的结果影响极小。

**警告**: 并不是小数值一定会得到好的结果。例如，**一个神经网络的层中的权重值很小，那么在反向传播的时候就会计算出非常小的梯度（因为梯度与权重值是成比例的）。这就会很大程度上减小反向传播中的“梯度信号”，在深度网络中，就会出现问题。**

**使用1/sqrt(n)校准方差**。**上面做法存在一个问题，随着输入数据量的增长，随机初始化的神经元的输出数据的分布中的方差也在增大。我们可以除以输入数据量的平方根来调整其数值范围，这样神经元输出的方差就归一化到1了。也就是说，建议将神经元的权重向量初始化为：**w = np.random.randn(n) / sqrt(n)。**其中**n是输入数据的数量。这样就保证了网络中所有神经元起始时有近似同样的输出分布。实践经验证明，这样做可以提高收敛的速度。

作者是He等人。文中给出了一种**针对ReLU**神经元的特殊初始化，并给出结论：网络中神经元的方差应该是![2.0/n](https://www.zhihu.com/equation?tex=2.0%2Fn)。代码为**w = np.random.randn(n) \* sqrt(2.0/n)**。这个形式是神经网络算法使用ReLU神经元时的当前最佳推荐。

**偏置（biases）的初始化。**通常将偏置初始化为0，这是因为随机小数值权重矩阵已经打破了对称性。对于ReLU非线性激活函数，有研究人员喜欢使用如0.01这样的小数值常量作为所有偏置的初始值，这是因为他们认为这样做能让所有的ReLU单元一开始就激活，这样就能保存并传播一些梯度。然而，这样做是不是总是能提高算法性能并不清楚（有时候实验结果反而显示性能更差），所以通常还是使用0来初始化偏置参数。

<font color=#B22222 >**实践: **当前的推荐是使用ReLU激活函数，并且使用**w = np.random.randn(n) \* sqrt(2.0/n)**来进行权重初始化，</font>

![image-20181003191222030](/Users/yunqingqi/Desktop/note/image-20181003191222030.png)

![image-20181003191246085](/Users/yunqingqi/Desktop/note/image-20181003191246085.png)

**批量归一化（Batch Normalization）**。[批量归一化](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1502.03167)是loffe和Szegedy最近才提出的方法，该方法减轻了如何合理初始化神经网络这个棘手问题带来的头痛：），<font color=#B22222 >其做法是让激活数据在训练开始前通过一个网络，网络处理数据使其服从标准高斯分布。(可以看上图理解)</font>因为归一化是一个简单可求导的操作，所以上述思路是可行的。在实现层面，应用这个技巧通常意味着全连接层（或者是卷积层，后续会讲）与激活函数之间添加一个BatchNorm层。对于这个技巧本节不会展开讲，因为上面的参考文献中已经讲得很清楚了，需要知道的是在神经网络中使用批量归一化已经变得非常常见。在实践中，使用了批量归一化的网络对于不好的初始值有更强的鲁棒性。最后一句话总结：批量归一化可以理解为在网络的每一层之前都做预处理，只是这种操作以另一种方式与网络集成在了一起。搞定！

<font color=#B22222 >**下面来具体了解一下 [Batch Normalization](https://zhuanlan.zhihu.com/p/34879333)** （在note里面也保存了这个页面，网站访问不了的话就可以去那里看）下面是重点总结：</font>

对于激活函数梯度饱和问题，有两种解决思路。第一种就是更为非饱和性激活函数，例如线性整流函数ReLU可以在**一定程度上**解决训练进入梯度饱和区的问题。**另一种思路是，我们可以让激活函数的<font color=#B22222 >输入分布保持在一个稳定状态</font>来尽可能避免它们陷入梯度饱和区，这也就是Normalization的思路。**

**Batch Normalization 的信号处理的解释： relu 是二极管， bn 就是电容器过滤掉直流成份，并控制增益不要超载**

在训练前，我们得检查一下loss是否合理，要设置training set 和 validation set

![image-20181005110422842](/Users/yunqingqi/Desktop/note/image-20181005110422842.png)

![image-20181005110432387](/Users/yunqingqi/Desktop/note/image-20181005110432387.png)

![image-20181005110457268](/Users/yunqingqi/Desktop/note/image-20181005110457268.png)

![image-20181005110511078](/Users/yunqingqi/Desktop/note/image-20181005110511078.png)

注意：我们在作小部分测试loss的时候，我们把 regularization 设置为0

**现在可以训练了**

![image-20181005110539592](/Users/yunqingqi/Desktop/note/image-20181005110539592.png)

此时regularization和learing rate开始设置的很小，从结果可以看到 loss not going down: learning rate too low ，learning rate很小，那么w梯度更新也很小，导致loss也很小

<font color=#B22222>但是我们看到，即使loss变化很小，training set的accuracy rate goes to 20% !!!, 这里的原因是：此时的probabilities are still pretty diffuse, so our loss term is still pretty similar, but when we shift all of these probabilities slightly in the right direction, because we're learning, now the accuracy all of a sunnden can jump, because we're taking the maximum correct value and so we're going to get a big jump in accuracy, even though our loss is still relatively diffuse</font>

![image-20181005173556591](/Users/yunqingqi/Desktop/note/image-20181005173556591.png)

可以看到，我们此时设置learning_rate 是10^6, 很大，导致cost：NaN,  所以我们要一直调试learning rate

### Hyperparameter Optimization

----

```
神经网络中可变化调整的因素很多:
神经网络结构: 层数, 每层神经元个数多少
初始化w和b的方法
Cost函数
Regularization: L1, L2
Sigmoid输出还是Softmax?
使用Droput?
训练集大小
mini-batch size
学习率(learning rate): η
Regularization parameter: λ
```

![image-20181005181625657](/Users/yunqingqi/Desktop/note/image-20181005181625657.png)

<font color=#B22222>**使用cross-validation**</font>

<font color=#B22222>**调试的时候，最好先从learning rate开始调试！！！！！！！！！**</font>

**从简单的出发: 开始实验**

如: MNIST数据集, 开始不知如何设置, 可以先简化使用0,1**两类图, 减少80%数据量**, 用**两层神经网络[784, 2] (比[784, 30, 2]快)**

更快的获取反馈: 之前每个epoch来检测准确率, 可以替换为**每1000个图**之后,（减少training set的数量）

​                           或者**减少validation set的量**, 比如用100代替10,000

结果刚开始肯定会不令人满意，我们也发现，**λ之前设置为1000, 因为减少了训练集的数量, λ为了保证weight decay一样,对应的减少λ = 20.0**

 也许结果学习率应该更低? =10 ，然后发现结果变好，假设保持其他参数不变: 30 epochs, mini-batch size: 10, λ=5.0 ，，实验学习率=0.025, 0.25, 2.5，当然了，我们可以画图来看看cost：（如果学习率太大, 可能造成越走越高, 跳过局部最低点太小, 学习可能太慢）

![image-20181005175803859](/Users/yunqingqi/Desktop/note/image-20181005175803859.png)

**对于学习率, 可以从0.001, 0.01, 0.1, 1, 10 开始尝试, 如果发现cost开始增大, 停止, 实验更小的微调**

对于MNIST, 先找到0.1, 然后0.5, 然后0.25

**对于提前停止学习的条件设置, 如果accuracy在**<font color=#B22222>**一段时间内变化很小 (不是一两次,是的是epoch)**</font>

之前一直使用学习率是常数, 可以开始设置大一下, 后面逐渐减少: 比如开始设定常数, 直到在验证集上准确率开始下降, 减少学习率 (/2, /3)

**对于regularization parameter λ:** 先不设定regularization, 要先把学习率调整好, 然后再开始实验λ, 1.0, 10, 100..., 找到合适的, 再微调

**对于mini-batch size:**

太小: 没有充分利用矩阵计算的library和硬件的整合的快速计算

太大: 更新权重和偏向不够频繁

好在mini-batch size和其他参数变化相对独立, 所以不用重新尝试,

**很好对一个方法就是用python调用 (grid search)包**

## 总结

训练一个神经网络需要：

- 利用小批量数据对实现进行梯度检查，还要注意各种错误。

- 进行合理性检查，确认初始损失值是合理的，在小数据集上能得到100%的准确率。

- 在训练时，跟踪损失函数值，训练集和验证集准确率，如果愿意，还可以跟踪更新的参数量相对于总参数量的比例（一般在1e-3左右），然后如果是对于卷积神经网络，可以将第一层的权重可视化。

- 推荐的两个更新方法是SGD+Nesterov动量方法，或者Adam方法。

- 随着训练进行学习率衰减。比如，在固定多少个周期后让学习率减半，或者当验证集准确率下降的时候。

- 使用随机搜索（不要用网格搜索）来搜索最优的超参数。分阶段从粗（比较宽的超参数范围训练1-5个周期）到细（窄范围训练很多个周期）地来搜索。

- 进行模型集成来获得额外的性能提高。



  **关于随机搜索和网格搜索： **网格搜索其实可以理解成暴力搜索，一般当超参数的数目稍小的时候，才会用网格搜索(**三四个（或者更少）的超参数**,当超参数的数量增长时，网格搜索的计算复杂度会呈现指数增长，这时要换用随机搜索)；随机搜索一般会根据超参数的边缘分布采样。 

  **以SVM为例，挑选SVM的超参数 C值、kernel类型和gamma值**。下面的配置表示我们要搜索两种网格：一种是linear kernel和不同C值；一种是RBF kernel以及不同的C和gamma值。Grid Search会挑选最适合的超参数值。

  ```text
  param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   ]
  ```



**[交叉熵（Cross-Entropy)](https://blog.csdn.net/rtygbwwwerr/article/details/50778098)**

**[Softmax函数与交叉熵](https://zhuanlan.zhihu.com/p/27223959)**

![image-20181005145229638](/Users/yunqingqi/Desktop/note/image-20181005145229638.png)

![image-20181005145931360](/Users/yunqingqi/Desktop/note/image-20181005145931360.png)

![image-20181005145954135](/Users/yunqingqi/Desktop/note/image-20181005145954135.png)

![image-20181005150033011](/Users/yunqingqi/Desktop/note/image-20181005150033011.png)

![image-20181005150047321](/Users/yunqingqi/Desktop/note/image-20181005150047321.png)

![image-20181005150208309](/Users/yunqingqi/Desktop/note/image-20181005150208309.png)

<font color=#B22222>**错误大，那么对于w的偏导大，更新多，学习的也就快**</font>

![image-20181005150632564](/Users/yunqingqi/Desktop/note/image-20181005150632564.png)

**这里的偏向指的是bias**

![image-20181005151006934](/Users/yunqingqi/Desktop/note/image-20181005151006934.png)

![image-20181005150524369](/Users/yunqingqi/Desktop/note/image-20181005150524369.png)

![image-20181005152000161](/Users/yunqingqi/Desktop/note/image-20181005152000161.png)

![image-20181005152238429](/Users/yunqingqi/Desktop/note/image-20181005152238429.png)

![image-20181005152401733](/Users/yunqingqi/Desktop/note/image-20181005152401733.png)

![image-20181005152434460](/Users/yunqingqi/Desktop/note/image-20181005152434460.png)

![image-20181005152806766](/Users/yunqingqi/Desktop/note/image-20181005152806766.png)



现在来看一个例子：

![image-20181005153227155](/Users/yunqingqi/Desktop/note/image-20181005153227155.png)

我们这个例子里，训练集只有1000，注意一点，下面要有对比

![image-20181005153323988](/Users/yunqingqi/Desktop/note/image-20181005153323988.png)

这个例子里，我们用的是50000个集合，如果我们不改变lmbda的话，图片里点那个式子的结果就会非常小，也就意味着w更新慢，我们要适当点加大lmbda,

 下面再来看看100个神经元的例子，和上面对比

![image-20181005153729697](/Users/yunqingqi/Desktop/note/image-20181005153729697.png)

再来看两个以前经常看到的overfitting的例子，

![image-20181005153924138](/Users/yunqingqi/Desktop/note/image-20181005153924138.png)

![image-20181005153932696](/Users/yunqingqi/Desktop/note/image-20181005153932696.png)

**下面就是原因**

![image-20181005154103905](/Users/yunqingqi/Desktop/note/image-20181005154103905.png)

<font color=#B22222>**重点就是：小的权重的情况下，x的一些随机变化不会对神经网络点模型造成太大影响，而且不会太受局部噪音的影响**</font>

现在来看看L1 和dropout

![image-20181005154834003](/Users/yunqingqi/Desktop/note/image-20181005154834003.png)

![image-20181005155200116](/Users/yunqingqi/Desktop/note/image-20181005155200116.png)

<font color=#B22222>**L1 和 L2  的不同就是，L1 减少的只是一个常量，L2减少的是一个固定的比例！仔细看图片里的结论**</font>

**随机失活（Dropout）**是一个简单又极其有效的正则化方法。该方法由Srivastava在论文[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/%257Ersalakhu/papers/srivastava14a.pdf)中提出的，与L1正则化，L2正则化和最大范式约束等方法互为补充。在训练的时候，随机失活的实现方法是让神经元以超参数![p](https://www.zhihu.com/equation?tex=p)的概率被激活或者被设置为0。<font color=#B22222>**（如果用了Batch normalization 的话，一般就不再用Dropout )**</font>

![image-20181005161340571](/Users/yunqingqi/Desktop/note/image-20181005161340571.png)

图片来源自[论文](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/%7Ersalakhu/papers/srivastava14a.pdf)，展示其核心思路。在训练过程中，随机失活可以被认为是对完整的神经网络抽样出一些子集，每次基于输入数据只更新子网络的参数（然而，数量巨大的子网络们并不是相互独立的，因为它们都共享参数）。在测试过程中不使用随机失活，可以理解为是对数量巨大的子网络们做了模型集成（model ensemble），以此来计算出一个平均的预测。

![image-20181005162632245](/Users/yunqingqi/Desktop/note/image-20181005162632245.png)

避免overfitting，提高准确率，还可以通过增大训练集，图片增强(旋转)来获得，用的时候搜一下**Data Augmentation**

但是图片增强时，要模拟真实世界中这种数据可能出现的变化，比如说：

![image-20181005163720692](/Users/yunqingqi/Desktop/note/image-20181005163720692.png)

旋转15度是ok的，但是你旋转180度的话就不太符号真实情况

![image-20181006123116912](/Users/yunqingqi/Desktop/note/image-20181006123116912.png)

<font color=#B22222>**现常用到方法就是只用Batch normalization,如果还是发现过拟合的话，就尝试L2或者dropout**</font>

<font color=#B22222>**现在再来回顾一下Batch normalization 和 cross-entropy: 导致sigmod 逼近饱和，我们其实是从两个方面进行改进的，就是Batch normalization 和 cross-entropy。 但是 cross-entropy 是从输出层改进的，Batch normalization 是从隐藏层改进的。 换句话说，cross-entropy  是对sigmod的输出结果改进的，而Batch normalization 对sigmod的输入改进的，看回顾一下两个算法的输入。**</font>

**cross-entropy 的输入：**![image-20181005165440125](/Users/yunqingqi/Desktop/note/image-20181005165440125.png)

可以看到 **cross-entropy 的输入** 是经过sigmod处理的，因为 **cross-entropy** 本来就是计算概率的差距的，输入的两个都是概率的分布

<font color=#B22222>**梯度下降算法的代替：(Tensorflow 上都有接口可以调用)**</font>

![image-20181005185119992](/Users/yunqingqi/Desktop/note/image-20181005185119992.png)

**Adam 的优势：**

```
直截了当地实现
高效的计算
所需内存少
梯度对角缩放的不变性（第二部分将给予证明）
适合解决含大规模数据和参数的优化问题
适用于非稳态（non-stationary）目标
适用于解决包含很高噪声或稀疏梯度的问题
超参数可以很直观地解释，并且基本上只需极少量的调参
```

我们通常推荐在深度学习模型中使用 Adam 算法或 SGD+Nesterov 动量法。(用的时候查一查各自的参数怎么调比较好)

**sigmoid导致vanishing gradient problem 的原因就是下面的例子： **

![image-20181006121336458](/Users/yunqingqi/Desktop/note/image-20181006121336458.png)

![image-20181006121410998](/Users/yunqingqi/Desktop/note/image-20181006121410998.png)

所以前面的隐藏层的learning rate会比每一层后面的隐藏层的learning rate要小，你的隐藏层越多的话，前面的隐藏层的learning rate就极其小，导致更新很慢，**而ReLU的导数为1，所以就avoid vanishing gradient problem** 

![image-20181006123716531](/Users/yunqingqi/Desktop/note/image-20181006123716531.png)

![image-20181006154853955](/Users/yunqingqi/Desktop/note/image-20181006154853955.png)

![image-20181006154912133](/Users/yunqingqi/Desktop/note/image-20181006154912133.png)

![image-20181006155130664](/Users/yunqingqi/Desktop/note/image-20181006155130664.png)

![image-20181006155203804](/Users/yunqingqi/Desktop/note/image-20181006155203804.png)



![image-20181006185031831](/Users/yunqingqi/Desktop/note/image-20181006185031831.png)

<font color=#B22222>**static的好处就是，一旦你serialize the graph，you have this data structure in memory that represents the entire structure of your network. And now you could take that data structure and just serialize it to disk. And now you've got the whole structure of your network saved in some file. And then you could later rear(培养，栽种，后面) load that thing and then run that computational graph without access to the original code that built it. You might imagine that you might want to train your in Python because it's maybe easier to work with, but then after you serialize that network and then you could deploy ot now in maybe a c++ environment where you don't need to use the original code to build the graph. So that's kind of a nice advantage of static graphs. **</font>

![image-20181006185956452](/Users/yunqingqi/Desktop/note/image-20181006185956452.png)

![image-20181006191719669](/Users/yunqingqi/Desktop/note/image-20181006191719669.png)

**如果想实现上面这样的结构，得用下面这样的方法**：

![image-20181006191748008](/Users/yunqingqi/Desktop/note/image-20181006191748008.png)

可以看出来，TensorFlow 几乎是把所有的功能用自己的语言重写了，而Pytorch 可以让我们使用python等其他语言。

![image-20181006192143070](/Users/yunqingqi/Desktop/note/image-20181006192143070.png)

![image-20181006192332603](/Users/yunqingqi/Desktop/note/image-20181006192332603.png)

**当我们想用dynamic graph 的时候，还是首选 Pytorch, 好像dynamic graph的创造性更大。。。。**

下面来看看几种神经网络的框架：

1. Alexnet

   pooling layer 是没有参数的，我们的参数就是weights

   ![image-20181007112645088](/Users/yunqingqi/Desktop/note/image-20181007112645088.png)

<font color=#B22222>**使用 3x3 的卷积核，相比于7x7的卷积核，会有更少的参数，更深，更非线性化**</font>

![image-20181007113958632](/Users/yunqingqi/Desktop/note/image-20181007113958632.png)

### [解释一下全连接层](https://zhuanlan.zhihu.com/p/33841176)（很详细！！！！！）

---

一般来说，卷积神经网络会有三种类型的隐藏层——**卷积层、池化层、全连接层**。卷积层和池化层比较好理解，主要很多教程也会解释。

- 卷积层(Convolutional layer)主要是用一个采样器从输入数据中采集关键数据内容；
- 池化层(Pooling layer)则是对卷积层结果的压缩得到更加重要的特征，同时还能有效控制过拟合。

在卷积神经网络的最后，往往会出现一两层全连接层，全连接一般会把卷积输出的二维特征图转化成一维的一个向量，这是怎么来的呢？目的何在呢？

举个例子：

![image-20181007114801484](/Users/yunqingqi/Desktop/note/image-20181007114801484.png)

最后的两列小圆球就是两个全连接层，在最后一层卷积结束后，进行了最后一次池化，输出了20个12x12的图像，然后通过了一个全连接层变成了1x100的向量。

这是怎么做到的呢，其实就是有20x100个12x12的卷积核卷积出来的，对于输入的每一张图，用了一个和图像一样大小的核卷积，这样整幅图就变成了一个数了，如果厚度是20就是那20个核卷积完了之后相加求和。这样就能把一张图高度浓缩成一个数了。

全连接的目的是什么呢？因为传统的网络我们的输出都是分类，也就是几个类别的概率甚至就是一个数--类别号，那么全连接层就是高度提纯的特征了，方便交给最后的分类器或者回归。

卷积层本来就是全连接的一种简化形式:不全连接+参数共享，同时还保留了空间位置信息。这样大大减少了参数并且使得训练变得可控。 全连接就是个矩阵乘法，相当于一个特征空间变换，可以把有用的信息提取整合。

**全连接层没有考虑到特征所在的空间位置**

![image-20181007121647080](/Users/yunqingqi/Desktop/note/image-20181007121647080.png)

![image-20181007121706273](/Users/yunqingqi/Desktop/note/image-20181007121706273.png)

**卷积层负责提取特征，采样层负责特征选择，全连接层负责分类**

![image-20181007151244449](/Users/yunqingqi/Desktop/note/image-20181007151244449.png)

注意：一个filter，一个activation map，所谓的low feature上面的笔记也有记，就是说边角，high level呢就是指有语义，比如眼睛，耳朵，鼻子，纽扣等等

![image-20181007160120647](/Users/yunqingqi/Desktop/note/image-20181007160120647.png)

![image-20181007171858872](/Users/yunqingqi/Desktop/note/image-20181007171858872.png)

![image-20181007172104081](/Users/yunqingqi/Desktop/note/image-20181007172104081.png)

![image-20181007172235097](/Users/yunqingqi/Desktop/note/image-20181007172235097.png)

![image-20181007172437565](/Users/yunqingqi/Desktop/note/image-20181007172437565.png)

![image-20181007172454619](/Users/yunqingqi/Desktop/note/image-20181007172454619.png)

![image-20181007180949866](/Users/yunqingqi/Desktop/note/image-20181007180949866.png)

![image-20181007181324293](/Users/yunqingqi/Desktop/note/image-20181007181324293.png)

![image-20181007183108544](/Users/yunqingqi/Desktop/note/image-20181007183108544.png)

![image-20181007204314040](/Users/yunqingqi/Desktop/note/image-20181007204314040.png)

[global average pooling](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)

[Resnet](https://www.jianshu.com/p/1c7668f1dc25)

[Resnet](https://segmentfault.com/a/1190000014421510)

Global average pooling 其实还是用到了一层全连接层，看下图：

![image-20181008110413601](/Users/yunqingqi/Desktop/note/image-20181008110413601.png)

最后有一个fc1000，还是要训练参数的～～～～～～～～～～～

下面看看  **循环神经网络，用于处理自然语言**

![image-20181008151321682](/Users/yunqingqi/Desktop/note/image-20181008151321682.png)

n 代表的是batch size，x 代表的是每个字的向量表达，而 h 就是说，我们事先确定隐含层向量的长度

我们认为 **y** 代表：下一个可能字符的数量的话，那么 $\hat{y}$ 的维度就是 $n * y$ , n 代表的是batch size，

所以我们现在要先确定隐含层向量的长度

**下面是卷积神经网络里的传播**

![image-20181008160234390](/Users/yunqingqi/Desktop/note/image-20181008160234390.png)

一定要注意 维度！！！！！  $\frac{\partial J}{\partial o}$ 求导后的维度一定和矩阵o的维度一样！！！！！！！

比较难的一个就是右上角再往下一个式子：$\frac{\partial J}{\partial z}$ , 两个导数相乘时，我们用的是element wise，因为 $\frac{\partial J}{\partial h}$ 的维度是 hx1, 而且 $\frac{\partial h}{\partial z}$ 的维度也是 hx1，所以得用元素相乘

<font color=#B22222>**下面我们来看一下 循环神经网络里的时序传播**</font>

---

![image-20181008163223276](/Users/yunqingqi/Desktop/note/image-20181008163223276.png)

<font color=#B22222>**一定要注意参数的流向！！！！！不懂了再会去看视频，第十三课！！！！**</font>

右下角的公式（通用公式），是对它左边和上边函数的总结，我们可以看到，当你到 T 很大，i 很小的时候，靠前的 $\frac{\partial L}{\partial h_t}$ 的导数就 explode（当w > 1）, 或者 vanish（当 w < 1）了，会出问题，所以我们会对 w 有个阈值

![image-20181008184001163](/Users/yunqingqi/Desktop/note/image-20181008184001163.png)

 **注意看data和label的关系，比如 9，10，11 它的label为10，11，12. 这是因为我们的输出就是下一个的输入！！**

![image-20181008193251477](/Users/yunqingqi/Desktop/note/image-20181008193251477.png)

我们用相邻的批量来train我们的神经网络

为了搞明白当我们用batchsize = 2 来训练到底是怎么运行的，我们让我们代码里的 rnn 函数参数 inputs 为：

```python
[[0,5],

 [1,6],

 [2,7],

 [3,8],

 [4,9]]
for X in inputs:
        print('X: ',X.shape)
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
上面这一段代码，每个 X的shape为[2,1027], 也就是说 H 的大小为[2,256],也就是[batch_size,number_hidden],也就是说 我们的权重 w_xh，w_hh, w_hy 在用 batch 方法训练的时候是共用的！！！！！，也就是说我们用batch的方法来训练，会加快训练呢！！！！
此时，我们 Y 的shape就是[2,1027],就是输入的 X 的结果， 
在 for 循环结束的时候，我们 outputs 的shape也就是 (5, (2, 1027)), 
在 下面这个预测函数里：
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx) # 这里batch_size 设为1
    output = [char_to_idx[prefix[0]]]   
#     print(prefix)  # 分开
#     print(prefix[0]) ## 分
#     print(output[-1]) # 130,也就是 分 这个字的 index
    for t in range(num_chars + len(prefix)):
        # 将上一时间步的输出作为当前时间步的输入。
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态。
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是 prefix 里的字符或者当前的最好预测字符。
        if t < len(prefix) - 1:    # 必须要 < len(prefix) - 1, 而不是 < len(prefix)
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar())) # 应该是输出最大值的index，用于下面的转换
    return ''.join([idx_to_char[i] for i in output])
```

可以看到，在整个训练的过程中，batch的作用就是加快训练我们的整个所有weight，weight的size不会根据batch变，只是输出会变，我们在预测的时候，只要把batch_size设置为1就可以预测一个字了，起作用的只是weight而已！！！！

循环神经网络中较容易出现梯度衰减或爆炸，在推导时间反向传播的时已经解释过了，为了防止它，我们用到了裁剪梯度 (grad_clipping)  :

![image-20181009140834990](/Users/yunqingqi/Desktop/note/image-20181009140834990.png)

让我们的梯度不超过 $\theta$

 语言模型里我们通常使用**困惑度（perplexity）**来评价模型的好坏

### [Epoch、Batch Size和迭代](https://www.jiqizhixin.com/articles/2017-09-25-3)(这篇将的很清楚)

### **[转置卷积](https://blog.csdn.net/LoseInVain/article/details/81098502)**



### SSD 物体检测

---

![image-20181016153241058](/Users/yunqingqi/Desktop/note/image-20181016153241058.png)

注意，我们一共需要画 n + m -1个框，因为 i <= n 和 i > n的时候，有一个是重复的，这里就要注意了，当 i <= n的时候，用的是 size[i] 和 ratios[0] 来锚框， 当 i > n的时候，用size[0] 和 ratios[i-n] 来锚框！！！！！，下面来看一个例子：

![image-20181016153925938](/Users/yunqingqi/Desktop/note/image-20181016153925938.png)

可以看到大小为 0.75 比例为 1 的蓝色锚框比较好的覆盖了图像中的小狗。

![image-20181016154312425](/Users/yunqingqi/Desktop/note/image-20181016154312425.png)

![image-20181016155345523](/Users/yunqingqi/Desktop/note/image-20181016155345523.png)

### [文章后面有作者自己的调参经验，很好](https://yq.aliyun.com/articles/610509)

### [NMS——非极大值抑制](https://blog.csdn.net/shuzfan/article/details/52711706) , 需要注意的是，Non-Maximum Suppression一次处理一个类别，如果有N个类别，Non-Maximum Suppression就需要执行N次。

### YOLOV3 的网络结构：

![image-20181018143818157](/Users/yunqingqi/Desktop/note/image-20181018143818157.png)

![image-20181018144100508](/Users/yunqingqi/Desktop/note/image-20181018144100508.png)

网络通过称为网络步幅的因子对图像进行下采样。例如，如果网络的步幅为32，则尺寸为416 x 416的输入图像将产生尺寸为13 x 13的输出。一般而言，网络中任何层的步幅等于该层的输出的尺寸比网络的输入图像的尺寸小的倍数

我们将输入图像划分成与**最终特征图**有**相同维度**的网格。

让我们思考下面的一个例子，输入图像是416 x 416，网络的步幅是32。 如前所述，特征图的维度将是13 x 13。于是，我们将输入图像划分为13 x 13个网格。

![image-20181019161754483](/Users/yunqingqi/Desktop/note/image-20181019161754483.png)

![image-20181019161812154](/Users/yunqingqi/Desktop/note/image-20181019161812154.png)

对于尺寸为416×416的图像，YOLO预测（（52×52）+（26×26）+ 13×13））×3 = 10647个边界框。**目标置信度的阈值处理**和 **非最大值抑制** 将会把10647个边界框缩小到几个

锚的尺寸根据*net*块的*height*和*width*属性。这些属性是输入图像的尺寸，它比检测图大（输入图像是检测图的*stride*倍）。因此，我们必须通过检测特征图的*stride*来划分锚。

```text
anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
```

OpenCV将图像加载为*numpy*数组，它的颜色通道顺序是BGR。 PyTorch的图像输入格式是（批x通道x高x宽），通道顺序为RGB。因此，我们在*util.py*中编写*prep_image*函数，将*numpy*数组转换为PyTorch的输入格式。

在我们编写这个函数之前，我们必须编写*letterbox_image*函数来调整图像的大小，保持宽高比一致，并用颜色（128,128,128）填充空白的区域。

### 下面来看一个样式迁移，好像目前应用于艺术和文字的模仿

![image-20181022145848278](/Users/yunqingqi/Desktop/note/image-20181022145848278.png)

<font color=#B22222 >对于样式层的选择，我们选择的是第一层和第三层，这是因为我们要想low-level 和 high-level 的特征，拿梵高的画为例，high-level的特征就是那些水彩，钩钩弯弯的painting，对于content，我们选取第二层的特征，因为我们不要求输出图像和原始图像完全一致，也不能不太像，所以就选择了中间层 </font>

![image-20181022145906695](/Users/yunqingqi/Desktop/note/image-20181022145906695.png)

![image-20181022151759677](/Users/yunqingqi/Desktop/note/image-20181022151759677.png)



![image-20181022152142575](/Users/yunqingqi/Desktop/note/image-20181022152142575.png)

使输入图片的像素为：均值为0，方差为1

### [样式迁移具体步骤](https://zh.gluon.ai/chapter_computer-vision/neural-style.html#%E6%95%B0%E6%8D%AE)

上面链接里，作者用的trick是：首先我们将图像调整到**高为 300 宽 200 来进行训练**，这样使得训练更加快速。合成**图像的初始值设成了内容图像**，使得初始值能尽可能接近训练输出来加速收敛。但是我们希望可以得到更加清晰的合成图像，所以作用又用到的trick：由于图像尺寸较小，所以细节上比较模糊。下面我们在**更大的 1200×800 的尺寸上训练**，希望可以得到更加清晰的合成图像。**为了加速收敛，我们将刚才训练到的（300*200）合成图像高宽放大 3 倍来作为初始值。**



SGD 之所以可以，是因为它取到batch来做计算，总体上它们到方向是朝着最小值去的，只是会绕一些弯而已，迭代的次数需要多一点，但是它节省了计算时间

老师最后还说，课后给的那个链接里的bacpropogation过程必须看，还有一本书！！

为什么 hinge loss 适用于 svm， cross entropy 适用于神经网络（老师说了什么bounded 和 unbounded，监督和无监督学习原因）

是这样：svm hinge loss 和  cross entropy 适用于 semisupervised learning， hinge loss适用于supervised learning（hinge loss 是unbounded）

sym hinge loss   适用于 semisupervised learning 是因为 semisupervised learning没有label，它不想让loss goes infinitely

我们只需要真的high level的意义就行，要能说出来 semisupervised learning 和 supervised learning 用的loss的区别在哪

![image-20181024161617192](/Users/yunqingqi/Desktop/note/image-20181024161617192.png)

上面这几个算法就是assign3 文章的内容，也就是manifold的内容

![image-20181025181558825](/Users/yunqingqi/Desktop/note/image-20181025181558825.png)

![image-20181109123129264](/Users/yunqingqi/Desktop/note/image-20181109123129264.png)

