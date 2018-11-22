[训练集的class不平衡](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

[Adaboost](https://blog.csdn.net/guyuealian/article/details/70995333)

Adaboost的其中一个思想就是说，如果一样样本m_1分错了，那么它的权值就会加大，这样的话，在下次计算error rate的时候：          如果我们依然把m_1分错的话，那么这样的分类器的error rate就很大，而我们选取的是error rate比较小的分类器，所以在寻找error rate小的分类器的过程中，就强迫那些分类器把m_1分对。。。。

[决策树](http://shiyanjun.cn/archives/417.html)

[决策树代码](https://segmentfault.com/a/1190000015083169)

[偏差和方差](https://www.jianshu.com/p/8d01ac406b40)1

[偏差和方差](https://blog.csdn.net/u010626937/article/details/74435109)2

#### 决策树、xgboost对缺失值不敏感，而SVM对缺失值比较敏感!!!

One-hot编码来处理类别！！！！！！！！！！！！！！！！

![image-20180918225846218](/Users/yunqingqi/Desktop/note/image-20180918225846218.png)

这里的x‘ 是说，针对x1这一列！！！！！进行0-1对scale，然后再单独对x2这一列进行0-1scale！！

而不是说真对每个sample，s1，所有的特征来做！！！！！

![image-20180918230057802](/Users/yunqingqi/Desktop/note/image-20180918230057802.png)

我们知道每个特征的最小值和最大值的时候，就用min-max方法，不知道的时候就用standardization



![image-20180918231523684](/Users/yunqingqi/Desktop/note/image-20180918231523684.png)

一般来说都是越往后的层，width越少

只要你有足够的数据量，越deep越好

事实上，瘦高的比矮胖的要好

![image-20180919100232616](/Users/yunqingqi/Desktop/note/image-20180919100232616.png)

![image-20180919100314447](/Users/yunqingqi/Desktop/note/image-20180919100314447.png)

sigmoid的会出现vanishing geadient，就是在接近0和1的时候，gradient都趋向于0

relu就不会出现vanishing geadient，也是最常用的

Tanh 通常用在：你的features range from negatives, NLP常用

![image-20180919102130601](/Users/yunqingqi/Desktop/note/image-20180919102130601.png)

分类不用squared error是因为，squared error 是非凸优化问题：

![image-20180919130256790](/Users/yunqingqi/Desktop/note/image-20180919130256790.png)



Pandas 读取文件

```python
d = pd.DataFrame(pd.read_csv("bitcoin.csv"))
read_csv("test.cvs",hearder=None) #不把第一行作列属性
```

Pandas 保存文件

```python
dataframe = pd.DataFrame({"Number of developers":a, 'Entropy': b,'Number of modified files':c,'Status':d})
dataframe.to_csv("parset.csv", index=False, sep=',')
```

Pandas 处理列

```python
data.icol(0)   #选取第一列
 ndev = d['ndev']   #也可以直接获取这一列
# .ix 用法
data = d.ix[:,'ndev'] # 选择ndev这一列的所有行
a = []
# 快速的将 ndev这一列所有的值 赋给 一个数组
for index,value in enumerate(data):
    a.append(ndev(data))


data.ix[[1,2],[0]]  #选择第2,3行第1列的值,  [1,2]指的是行数，[0]指的是列
Out[15]: 
        a
two     5
three  10

data.ix[1:3,[0,2]]  #选择第2-4行第1、3列的值
Out[17]: 
        a   c
two     5   7
three  10  12
```

Pandas 替换一列中的值

```python
df['TermIndex']=df['TermIndex'].replace([1,2],['一','二']) #替换“TermIndex”的值，将数字转为中文
# 可以将sex这一列转成0，1格式，并设置为int类型, 个人觉得这样更好！～！～！～！～！
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
```

Pandas 一个完整的例子：

```python
#-*-coding:utf-8 -*-
import numpy as np
import pandas as pd
data = pd.DataFrame(pd.read_csv('letters_CG.csv'))
data['Class']=data['Class'].replace(['C','G'],[1,-1]) #替换“TermIndex”的值，将数字转为中文
data.to_csv('letters_CG.csv',index=False, encoding='utf-8')
```



python 替换列表中的元素

```python
a=[1,2,3,4,5,1,2,3,4,5,1]
>>> for n,i in enumerate(a):
...   if i==1:
...      a[n]=10
```

pandas DataFrame数据转为list

```python
data_x = pd.read_csv("E:/Tianchi/result/features.csv",usecols=[2,3,4])#pd.dataframe
data_y =  pd.read_csv("E:/Tianchi/result/features.csv",usecols=[5])

train_data = np.array(data_x)#np.ndarray()
train_x_list=train_data.tolist()#list
```

Pandas 删除列

```python
data.drop(['winter'],axis=1,inplace=True)
```

python 里的len(matrix)，输出的是matrix的行数

w 必须用w = cvx.Variable(len(tsample), 1) 来定义，因为我们是没办法给w初始值的，cvx.Variable来定义就好了

Pandans 转成np.array用 DataFrame.values

Matlab里向量的追加

```matlab
r1 = [ 1 2 3 4 ];
r2 = [5 6 7 8 ];
r = [r1,r2]    //追加到行
rMat = [r1;r2]   //追加到列
 
c1 = [ 1; 2; 3; 4 ];
c2 = [5; 6; 7; 8 ];
c = [c1; c2]
cMat = [c1,c2]
```

Numpy.dot 同线性代数中矩阵乘法的定义,就是一个2x3的矩阵和3x2的矩阵，得到2x2矩阵

 Numpy里的*就代表着对应元素相乘

python里的 ** 代表：几次幂

```python
a = 2
b = 3
c = a**b //结果是8
```

numpy中arange的使用方法

```python
>>> np.arange(3)
array([0, 1, 2])

>>> np.arange(1,3,0.3)
array([ 1. ,  1.3,  1.6,  1.9,  2.2,  2.5,  2.8])

>>> np.arange(1,12,2)
array([ 1,  3,  5,  7,  9, 11])
```

python里的[::-1]使用:  list[::-1]是将列表反过来

```python
比如说我有一个list = [1,2,3,4,5,6,7,7,8] 我想访问从倒数第一位到倒数第三位怎么做到? 我想要的输出效果应该是[8,7,7]

>>> list = [1,2,3,4,5,6,7,7,8]
>>> list[::-1][:3]
[8, 7, 7]
>>> list[-3:][::-1]
[8, 7, 7]
```

np.random.randint 用法：

![image-20180827184037381](/Users/yunqingqi/Desktop/note/image-20180827184037381.png)

np.random.choice 用法：

```python
# 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
a1 = np.random.choice(a=5, size=3, replace=False, p=None)
print(a1)
# 非一致的分布，会以多少的概率提出来
a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
print(a2)
# replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
```

np.min使用

```python
a = np.array([[1,5,3],[4,2,6]])
print(a.min()) #无参，所有中的最小值
print(a.min(0)) # axis=0; 每列的最小值
print(a.min(1)) # axis=1；每行的最小值
结果：
1
[1 2 3]
[1 2]
```

python里用astype(np.bool)来把一个数组里的值赋给另外一个数组

```py
import numpy as np
pop = np.random.randint(2, size=(200, 10))
parent = pop[0]
print("parent b4:",parent)


i_ = np.random.randint(0, 200, size=1)
print("pop[i_]",pop[i_])

cross_points = np.random.randint(0, 2, size=10).astype(np.bool)

print("cross_points:",cross_points)

parent[cross_points] = pop[i_, cross_points]
print("pop[i_, cross_points]",pop[i_, cross_points])

print("parent after:",parent)
```

而 DNA 呢, 可以都用数字, 而且可以用 [ASCII 编码](http://www.asciitable.com/). 将数字转化成字符, 或者字符转数字都可以, 我们为了统一, DNA 都用数字形式.

```
class GA:
    def translateDNA(self, DNA):    # convert to readable string
        return DNA.tostring().decode('ascii')
```

而字符转数字可以用 numpy 的这个功能:

```
>>> np.fromstring('dasd@', dtype=np.uint8)
# array([100,  97, 115, 100,  64], dtype=uint8)
```

若b之前是4x3的一个数组，我们用b.shape = 2, -1,那么b的shape就会是2x6了，-1的作用是自动计算此维度的长度

使用reshape方法，可以创建改变了尺寸的新数组，原数组的shape保持不变

```python
c = b.reshape((4, -1))
print "b = \n", b     ／／b还是3x4
print 'c = \n', c     ／／c是4x3
# 数组b和c共享内存，修改任意一个将影响另外一个
b[0][1] = 20
print "b = \n", b
print "c = \n", c
```

改变np数组的数据类型，应该是astype来修改

```python
d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
# f = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.complex)
# # 如果更改元素类型，可以使用astype安全的转换
f = d.astype(np.int)
print f
```

果生成一定规则的数据，可以使用NumPy提供的专门函数

```python
# arange函数类似于python的range函数：指定起始值、终止值和步长来创建数组
# 和Python的range类似，arange同样不包括终值；但arange可以生成浮点类型，而range只能是整数类型
# arange 默认是 步长为1 ， 初始值为0，重点值是输入的，然后个数就按步长来
a = np.arange(1, 10, 0.5)
print a
结果是：
[1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5 9.  9.5]
```

np可以设置打印一行显示多长的数组，有时候一行很长，他会换行或者给你省略

```python
np.set_printoptions(linewidth = 200)
```

linspace函数通过指定起始值、终止值和元素个数来创建数组，缺省包括终止值

```python
  #可以把linspace理解为是等差数列
   b = np.linspace(1, 10, 10)
   print ('b = ', b)
   结果就是：
   b =  [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
# 可以通过endpoint关键字指定是否包括终值
c = np.linspace(1, 10, 10, endpoint=False)
print （'c = ', c）
结果就是：
c =  [1.  1.9 2.8 3.7 4.6 5.5 6.4 7.3 8.2 9.1]
```

 和linspace类似，logspace可以创建等比数列

```python
# 下面函数创建起始值为10^1，终止值为10^2，有10个数的等比数列
d = np.logspace(1, 2, 9, endpoint=True)
print d
# 结果是 ：由于没有设定一行显示多少就会自动换行，看着不舒服
[ 10.          13.33521432  17.7827941   23.71373706  31.6227766
  42.16965034  56.23413252  74.98942093 100.        ]

#下面创建起始值为2^0，终止值为2^10(包括)，有10个数的等比数列
f = np.logspace(0, 10, 11, endpoint=True, base=2)
print f
```

使用 frombuffer, fromstring, fromfile等函数可以从字节序列创建数组

```python
    s = 'abcdz'
    g = np.fromstring(s, dtype=np.int8)
    print g
    #结果是 ：
    [ 97  98  99 100 122]
```

python的切片

```python
    a = np.arange(10)
    # 步长为2
    print (a[1:9:2])
    # 结果就是：（不包括9）
    [1 3 5 7]
    # 步长为-1，即翻转
    print （a[::-1]）
    # 结果就是：
    [9 8 7 6 5 4 3 2 1 0]
```

如果用了a = b[2:5]，并且之后了a或b其中一个，那么a和b都会被修改，一定要注意！！！！！！！！！！！！！！！！！

```python
    # 根据整数数组存取：当使用整数序列对数组元素进行存取时，
    # 将使用整数序列中的每个元素作为下标，整数序列可以是列表(list)或者数组(ndarray)。
    # 使用整数序列作为下标获得的数组不和原始数组共享数据空间。
    a = np.logspace(0, 9, 10, base=2)
    print (a)
    i = np.arange(0, 10, 2)
    print (i)
    # 利用i取a中的元素
    b = a[i]
    print (b)
    # b的元素更改，a中元素不受影响
    b[2] = 1.6
    print (b)
    print (a)
    结果是：
[  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
[0 2 4 6 8]
[  1.   4.  16.  64. 256.]
[  1.    4.    1.6  64.  256. ]
[  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```

```python
# 使用布尔数组i作为下标存取数组a中的元素：返回数组a中所有在数组b中对应下标为True的元素
# 生成10个满足[0,1)中均匀分布的随机数
a = np.random.rand(10)
print a
# # 大于0.5的元素索引
print a > 0.5
# # 大于0.5的元素
b = a[a > 0.5]        // 这就和之前那个基因算法里用布尔值来取数一样的
print b
# # 将原数组中大于0.5的元素截取成0.5
a[a > 0.5] = 0.5
print a
# # # # b不受影响
print b
```

现在来看一个广播实例：

```python
a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
列向量和行向量相加，列向量会被广播，结果是：
[[ 0  1  2  3  4  5]
 [10 11 12 13 14 15]
 [20 21 22 23 24 25]
 [30 31 32 33 34 35]
 [40 41 42 43 44 45]
 [50 51 52 53 54 55]]
# # 二维数组的切片
print (a[[0, 1, 2], [2, 3, 4]])  // [ 2 13 24] 第0行第二列，第1行第三列，第2行第四列
print (a[4, [2, 3, 4]])      //[42 43 44] 第4行， 2 3 4 列
print (a[4:, [2, 3, 4]])     // 第四行开始到最后一行，并取他们的 2 3 4 列，结果是：
                                                                [[42 43 44]
                                                                 [52 53 54]]
i = np.array([True, False, True, False, False, True])    
print (a[i])    // 是按行来的 ，所以结果是：
						[[ 0  1  2  3  4  5]
                         [20 21 22 23 24 25]
                         [50 51 52 53 54 55]]
print (a[i, 3])    // [ 3 23 53] 
```

   python 元素去重(利用python里的set)

```python
    # 4.2.1直接使用库函数
    a = np.array((1, 2, 3, 4, 5, 5, 7, 3, 2, 2, 8, 8))
    print ('原始数组：', a)
    # # 使用库函数unique
    b = np.unique(a)
    print ('去重后：', b)     //  [1 2 3 4 5 7 8]
    # # 4.2.2 二维数组的去重，结果会是预期的么？
    c = np.array(((1, 2), (3, 4), (5, 6), (1, 3), (3, 4), (7, 6)))
    print (u'二维数组：\n', c)
    print ('去重后：', np.unique(c)) //因为unique的工作原理就是把数组拉成一维的再去重，所以结果是：
    									[1 2 3 4 5 6 7]
    # 4.2.3 方案2：利用set
    print ('去重方案2：\n', np.array(list(set([tuple(t) for t in c]))))
```

np.stack 和 axis:

```python
    a = np.arange(1, 10).reshape((3, 3))
    b = np.arange(11, 20).reshape((3, 3))
    c = np.arange(101, 110).reshape((3, 3))
    print ('a = \n', a)
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    print ('b = \n', b)
     [[11 12 13]
     [14 15 16]
     [17 18 19]]
    print ('c = \n', c)
     [[101 102 103]
     [104 105 106]
     [107 108 109]]
    print ('axis = 0 \n', np.stack((a, b, c), axis=0))
    
   	[[[  1   2   3]       # axis = 0 就是说每个元素(a,b,c)堆叠起来
      [  4   5   6]
      [  7   8   9]]

     [[ 11  12  13]
      [ 14  15  16]
      [ 17  18  19]]

     [[101 102 103]
      [104 105 106]
      [107 108 109]]]
    print ('axis = 1 \n', np.stack((a, b, c), axis=1))
     [[[  1   2   3]      # axis = 1 就是说每个元素(a,b,c)的
      [ 11  12  13]							
      [101 102 103]]

     [[  4   5   6]					#     第二行堆叠起来
      [ 14  15  16]
      [104 105 106]]

     [[  7   8   9]					#     第三行堆叠起来
      [ 17  18  19]
      [107 108 109]]]   
    print ('axis = 2 \n', np.stack((a, b, c), axis=2))
     [[[  1  11 101]     	# axis = 2 就是说每个元素(a,b,c)的 每个一个位置堆叠起来
      [  2  12 102]
      [  3  13 103]]

     [[  4  14 104]
      [  5  15 105]
      [  6  16 106]]

     [[  7  17 107]
      [  8  18 108]
      [  9  19 109]]]
    
```

np.dot 和 * 区别：

```python
    a = np.arange(1, 10).reshape(3,3)
    print a
    b = a + 10
    print b
    print np.dot(a, b)   // dot 是就是一行和一列相乘
    print a * b          // * 就是相应元素相乘
    [[ 11  24  39]
     [ 56  75  96]
     [119 144 171]]
```

```python
a = np.arange(1, 10)
print a
b = np.arange(20,25)
print b
print np.concatenate((a, b))    //连接a和b
```

绘图

```python
# 是设置了字体，让其能显示中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']  #FangSong/黑体 FangSong/KaiTi
# 让 符号（ - ）能显示出来
mpl.rcParams['axes.unicode_minus'] = False

mu = 0
sigma = 1
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 51)
y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
plt.figure(facecolor='w')
#. 第一图是red线，       第二个图是green的圆圈，圆圈大小是8
plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
# plt.plot(x, y, 'ro-', linewidth=2)  // 上面的那句话页可以这样简写，就是说图片里有red线和圈
# x 轴
plt.xlabel('X', fontsize=15)
# y 轴
plt.ylabel('Y', fontsize=15)
# 图片名称
plt.title(u'高斯分布函数', fontsize=18)
# 是否带格子栏
plt.grid(True)
plt.show()
```

另外一个图

```python
x = np.linspace(start=-2, stop=3, num=1001, dtype=np.float)
y_logit = np.log(1 + np.exp(-x)) / math.log(2)
y_boost = np.exp(-x)
y_01 = x < 0
y_hinge = 1.0 - x
y_hinge[y_hinge < 0] = 0
# label 就是说在图里添加legend
# 还 有bar 图
plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
# -- 代表虚线
plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
plt.grid()
# label的位置   这个必须写！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
plt.legend(loc='upper right')
# 还可以保存下来
plt.savefig('1.png')
plt.show()
```

```
plt.subplot(121) 就是说我一行要画两个图，最后一个1 代表的是我心中画的是第一个图

```

下面介绍绘制三维图像：

```python
    #x, y = np.ogrid[-3:3:100j, -3:3:100j]
    u = np.linspace(-3, 3, 101)
	x, y = np.meshgrid(u, u)
    z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
    # z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0.1)  #
    ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.Accent, linewidth=0.5)
    plt.show()
    # cmaps = [('Perceptually Uniform Sequential',
    #           ['viridis', 'inferno', 'plasma', 'magma']),
    #          ('Sequential', ['Blues', 'BuGn', 'BuPu',
    #                          'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
    #                          'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
    #                          'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
    #          ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
    #                              'copper', 'gist_heat', 'gray', 'hot',
    #                              'pink', 'spring', 'summer', 'winter']),
    #          ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
    #                         'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
    #                         'seismic']),
    #          ('Qualitative', ['Accent', 'Dark2', 'Paired', 'Pastel1',
    #                           'Pastel2', 'Set1', 'Set2', 'Set3']),
    #          ('Miscellaneous', ['gist_earth', 'terrain', 'ocean', 'gist_stern',
    #                             'brg', 'CMRmap', 'cubehelix',
    #                             'gnuplot', 'gnuplot2', 'gist_ncar',
    #                             'nipy_spectral', 'jet', 'rainbow',
    #                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
```

Pandas :

```python
    pd.set_option('display.width', 200)  // 和 numpy 一样，可以设置一行显示的长度
    data['total'] = data['Jan'] + data['Feb'] + data['Mar']  //可以自动加一列total数据
    // 可以获得最小值，最大值，平均值，和
   	print (data['Jan'].sum())
    print (data['Jan'].min())
    print (data['Jan'].max())
    print (data['Jan'].mean())
   
	// 可以将sex这一列转成0，1格式，并设置为int类型
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
```

from sklearn.model_selection import train_test_split 来做cross validation

```python
data = pd.read_csv('Advertising.csv')    # TV、Radio、Newspaper、Sales
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
# train_size 就是说有多少用来train 
# 如果你想让每次的训练集不一样的话，就把 random_state=1 去掉，random_state 就是random_seed，只要你给了，那么每次训练集就是一样的！！！！！！！！！！！！！！！！！！！
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.8)

# 下面这个方法更适用于把csv里的特征和label分开
y = data['label'].values
x = data.values[:, 1:]            # 注意是.values[:,1:]
```

Numpy.set_printoptions(suppress = True) 可以让输出结果不用e的多少次方显示

from sklearn.model_selection import GridSearchCV 可以用来让这个库自己帮我们选好的超参数

```python
from sklearn.model_selection import GridSearchCV

model = Ridge()
alpha_can = np.logspace(-3, 2, 10)
np.set_printoptions(suppress=True)
# 会从我们给的范围，自动喂给model
lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
# 这是选择一个fit，即合适的alpha
lasso_model.fit(x_train, y_train)
print ('超参数：\n', lasso_model.best_params_)
```

有时候为了作图好看，需要这样做

```python
# 让y的真实值递增 argsort 的作用是返回一个order，这样画图的时候，就会比不这样做的时候方便看
order = y_test.argsort(axis=0)
y_test = y_test.values[order]
x_test = x_test.values[order, :]
```

在使用pandas读取多列的时候

```python
data = pd.read_csv(path, header=None)
# np.arange(4) 和range(4)的作用是一样的，但是就是读取不了数据，因为我只要print(range(4)),返回的是range(0,4)，而不是明显的array，就不行
x, y = data[np.arange(4)], data[4]
# Categorical(y).codes 就是把y那一列中的值分类了
y = pd.Categorical(y).codes
```

计算错误率或者准确率时，可以用True，然后计算sum比较简单

```python
float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)
还可以专门写个函数：(tip是哪个方法，比如XGboost,随机森林)
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print ('%s正确率：%.3f%%' % (tip, acc_rate))
    return acc_rate
```

pandas里判断某一列的值是否缺失（或者是否为0）

```python
# 如果 isnull 或者 = 0 的个数大于0，就说明存在船票缺失
if len(data.Fare[data.Fare == 0]) > 0:  # data.Fare == 0 返回的值是index 和 bool
# 完整的代码在这：
if len(data.Fare[data.Fare == 0]) > 0:
    fare = np.zeros(3)
    # 让fare 代表 1 2 3 等舱的 均价！！！！！
    for f in range(0, 3):
        # .dropna() 的作用就是说我选择取 1 等舱的所有票价，然后drop掉无效的值(缺失，NAN)，然后取平均
        fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare == 0) & (data.Pclass == f + 1), 'Fare'] = fare[f]
```

   Pandas.get_dummies 的使用：

```python
embarked_data = pd.get_dummies(data.Embarked)
embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
# concat 里 axis=1 就是在列上加新数据
data = pd.concat([data, embarked_data], axis=1)
data.to_csv('New_Data.csv')
```

比如 Embarked 里有 4个不相关的类别：C  Q  S  U，我们想把它们每一个都当一个新的列:下面的图就是结果

![image-20180906154915742](/Users/yunqingqi/Desktop/note/image-20180906154915742.png)

numpy.tile 复制

```python
arr = np.array([1,2])
# [1 2]
arr_ = np.tile(arr, 2)
# [1 2 1 2]
arr__ = np.tile(arr, (3,2))  # 最后一个2，是说按行来说，要复制几次
							 # 前面一个3，就是按列来说，要复制几次
[[1 2 1 2]
 [1 2 1 2]
 [1 2 1 2]]

```

from sklearn.metrics import accuracy_score 用法

```python
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
xgb_acc = accuracy_score(y_test, y_hat)
# 还有一个是这样的 clf.score也可以，但是他们各自的参数不同，要注意
# 准确率
clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
print (clf.score(x_train, y_train)) # 精度
print ('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
print (clf.score(x_test, y_test))
print ('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
```

svm里面的decision_function()

```python
# decision_function 返回的是三个值，这些值就是一个数据点，到3个分类超平面的距离！！！！哪个值大就选哪个(即属于哪个分类)
print ('decision_function:\n', clf.decision_function(x_train))
```

创建矩阵的一些小技巧，还是用的reshpe和tile

```python
a = np.array((0,1,2,3)).reshape((-1, 1))
y = np.tile(a, N).flatten()
# 结果就是：
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
```

替换

![image-20180907172436822](/Users/yunqingqi/Desktop/note/image-20180907172436822.png)





### 下面的都是gluon的计算：（和numpy差不多）

有些情况下，我们需要随机生成 NDArray 中每个元素的值。下面我们创建一个形状为（3，4）的 NDArray。它的每个元素都随机采样于均值为 0 标准差为 1 的正态分布。

```
x = [[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
 
y= [[2. 1. 4. 3.]
 [1. 2. 3. 4.]
 [4. 3. 2. 1.]]
```

```
nd.random.normal(0, 1, shape=(3, 4))
```

```
[[ 2.2122064   0.7740038   1.0434405   1.1839255 ]
 [ 1.8917114  -1.2347414  -1.771029   -0.45138445]
 [ 0.57938355 -1.856082   -1.9768796  -0.20801921]]
<NDArray 3x4 @cpu(0)>
```

按元素乘法：x * y

按元素除法：x / y

按元素做指数运算： y.exp()

除了按元素计算外，我们还可以使用`dot`函数做矩阵运算。下面将`x`与`y`的转置做矩阵乘法。由于`x`是 3 行 4 列的矩阵，`y`转置为 4 行 3 列的矩阵，两个矩阵相乘得到 3 行 3 列的矩阵。

```
nd.dot(x, y.T)
```

我们也可以将多个 NDArray 合并。下面分别在行上（维度 0，即形状中的最左边元素）和列上（维度 1，即形状中左起第二个元素）连结（concatenate）两个矩阵。

```
nd.concat(x, y, dim=0), nd.concat(x, y, dim=1)
(
 [[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]
  [ 2.  1.  4.  3.]
  [ 1.  2.  3.  4.]
  [ 4.  3.  2.  1.]]
 <NDArray 6x4 @cpu(0)>,
 [[ 0.  1.  2.  3.  2.  1.  4.  3.]
  [ 4.  5.  6.  7.  1.  2.  3.  4.]
  [ 8.  9. 10. 11.  4.  3.  2.  1.]]
 <NDArray 3x8 @cpu(0)>)
```

使用条件判断式可以得到元素为 0 或 1 的新的 NDArray。以`x == y`为例，如果`x`和`y`在相同位置的条件判断为真（值相等），那么新的 NDArray 在相同位置的值为 1；反之为 0。 (和numpy一样)

```
x == y
[[0. 1. 0. 1.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

我们可以通过`asscalar`函数将结果变换为 Python 中的标量。下面例子中`x`的 L2L2 范数结果同上例一样是单元素 NDArray，但最后结果变换成了 Python 中标量。

```
x.norm().asscalar()
22.494444
```

我们也可以把`y.exp()`、`x.sum()`、`x.norm()`等分别改写为`nd.exp(y)`、`nd.sum(x)`、`nd.norm(x)`等。

## 索引

在下面的例子中，我们指定了 NDArray 的行索引截取范围`[1:3]`。依据左闭右开指定范围的惯例，它截取了矩阵`x`中行索引为 1 和 2 的两行。

```
x[1:3]
[[ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
```

我们可以指定 NDArray 中需要访问的单个元素的位置，例如矩阵中行和列的索引，并为该元素重新赋值。

```
x[1, 2] = 9
[[ 0.  1.  2.  3.]
 [ 4.  5.  9.  7.]
 [ 8.  9. 10. 11.]]
```

当然，我们也可以截取一部分元素，并为它们重新赋值。下面例子中，我们为行索引为 1 的每一列元素重新赋值。

```
x[1:2, :] = 12
[[ 0.  1.  2.  3.]
 [12. 12. 12. 12.]
 [ 8.  9. 10. 11.]]
```

## 运算的内存开销

前面例子里我们对每个操作新开内存来储存运算结果。举个例子，即使像`y = x + y`这样的运算，我们也会新创建内存，然后将`y`指向新内存。为了演示这一点，我们可以使用 Python 自带的`id`函数：如果两个实例的 ID 一致，那么它们所对应的内存地址相同；反之则不同。

```
before = id(y)
y = y + x
id(y) == before     ／／False
```

如果我们想指定结果到特定内存，我们可以使用前面介绍的索引来进行替换操作。在下面的例子中，我们先通过`zeros_like`创建和`y`形状相同且元素为 0 的 NDArray，记为`z`。接下来，我们把`x + y`的结果通过`[:]`写进`z`所对应的内存中。

```
z = y.zeros_like()
before = id(z)
z[:] = x + y
id(z) == before    ／／True
```

实际上，上例中我们还是为`x + y`创建了临时内存来存储计算结果，再复制到`z`所对应的内存。如果想避免这个临时内存开销，我们可以使用运算符全名函数中的`out`参数。

```
nd.elemwise_add(x, y, out=z)
id(z) == before    ／／True
```

如果`x`的值在之后的程序中不会复用，我们也可以用 `x[:] = x + y` 或者 `x += y` 来减少运算的内存开销。

```
before = id(x)
x += y
id(x) == before    ／／ True
```

## NDArray 和 NumPy 相互变换

我们可以通过`array`和`asnumpy`函数令数据在 NDArray 和 NumPy 格式之间相互变换。下面将 NumPy 实例变换成 NDArray 实例。

```
import numpy as np

p = np.ones((2, 3))
d = nd.array(p)
d
```

```
[[1. 1. 1.]
 [1. 1. 1.]]
```

再将 NDArray 实例变换成 NumPy 实例。

```
d.asnumpy()
```

```
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
```

matlab 里 cell函数用法

```matlab
%这时就用到了cell数据类型了。cell的每个单元都可以存储任何数据，比如传递函数等。当然，存储矩阵更是没有问题的了。但是用cell数据类型之前，要先初始化。
a=cell(n,m)
% 那么就把a初始化为一个n行m列的空cell类型数据。
如何赋值呢？
a{1,1}=rand(5)
那么a的1行1列的单元中存储的就是一个随机的5×5的方阵了。
那么要用第一个单元中的方阵中的某个值呢？
可以如下引用：a{1,1}(2,3)
就可以了，引用cell单元时要用{},再引用矩阵的某个数据就要用()了。
cell单元中的每个单元都是独立的，可以分别存储不同大小的矩阵或不同类型的数据。
下面举个例子：
a=cell(2,2);%预分配
a{1,1}='cellclass';
a{1,2}=[1 2 2];
a{2,1}=['a','b','c'];
a{2,2}=[9 5 6];
>> a{1,1}
ans =cellclass
>> a{1,2}
ans =     1     2     2
>> a{2,:}
ans =abcans =     9     5     6
>> b=a{1,1}
b =cellclass
```

Numpy 数组的初始化

```python
werr = np.zeros((size,1),dtype=np.float)
a = np.array([2,3,4])
```

python寻找list中最大值、最小值并返回其所在位置

```python
c = [-10,-5,0,5,3,10,15,-20,25]
print c.index(min(c))  # 返回最小值
print c.index(max(c)) # 返回最大值
```

Numpy中找出array中 最大值/最小值 所对应的行和列

```python
np.max(a) 返回 a 里的最大值

# 第一种方法就是把np.array转成python的list，然后再用上面的方法
a= np.array([9, 12, 88, 14, 25])
list_a = a.tolist()
# 第二种方法就使用np自带的where
见下面的图片
# 还有就是找 行 和 列 最小
c = np.array([[11, 2, 8, 4], [4, 52, 6, 17], [2, 8, 9, 100]])
print(np.argmin(c, axis=0)) # 按每列求出最小值的索引
print(np.argmin(c, axis=1)) # 按每行求出最小值的索引

#### 第二种方法 返回的是tuple类型的数据，得用index 比如：[0] 来获取
#### 第三种 返回的是类型是'numpy.ndarray'，所以得用[0]这样来获取

```

where的方法获取index ![image-20180927164644522](/Users/yunqingqi/Desktop/note/image-20180927164644522.png)

numpy 数组从小到大排序后的索引值,和数组

```python
一维数组排序
x = np.array([3, 1, 2])
# 按升序排列
sorted_x_index = np.argsort(x)
# array([1, 2, 0])
# 按降序排列
sorted_x_index  = np.argsort(-x) 
# array([0, 2, 1])

# 得到argsort排序后的数组
sorted_x = x[np.argsort(x)]

# numpy 里把数组倒序，因为上面的 sorted_x_index = np.argsort(x) 的类型就是array,也可以用[::-1]的形式倒序
sorted_x_index[::-1] # 就可以了
```

matlab之.* 点乘和 *乘

```python
1、a*b就是矩阵乘法

2、a.*b就是a,b的对应元素相乘
```

python .* 点乘和 *乘

```python

import numpy
a = numpy.array([[1,2],
                 [3,4]])
b = numpy.array([[5,6],
                 [7,8]])
a*b     # a*b就是a,b的对应元素相乘(element-wise product)或者 np.multiply()
>>>array([[ 5, 12],
          [21, 32]])

numpy.dot(a,b)  # np.dot 就是矩阵乘法
>>>array([[19, 22],
          [43, 50]])
```

Python 里 的numpy也好，python自带的array也好，一定要注意

```python
x[0:-1]    #返回的并不是所有元素
x[0:]      #返回的才是所有元素
```

Python 里 计算一个 列表 里某个元素出现的次数

```python
# 注意是列表才可以用
>>> x = [1,2,'a',[1,2],[1,2]]
>>> x.count([1,2])
2
>>> x.count(1)
1
>>> x.count('a')
1
```

python 里计算一个 数组(python自带的还有numpy都可以) 里某个元素出现的次数

```python
import numpy as np
a = np.ones((4,5))
print(a)
print(np.sum(a==1))
```

list, dict, numpy.ndarray, dataframe数据格式如何转换？

```python


1. list转化为numpy.ndarray：
np.array(example)

2. numpy.ndarray转化为list：
list(example)

3. dict转化为dataframe:
    example['a'] = {'bb':2, 'cc':3}
    eee = pd.DataFrame(example)

4. numpy.ndarray转化为dataframe:
pd.DataFrame(example)

5. dataframe转化为numpy.ndarray：
example.values[:, :]

```

初始化权重

```python
D = np.ones((m,1),float)/m
```

Python替换NumPy数组中大于某个值的所有元素，用于计算分类器的error rate

```python
arr[arr > 255] = x
####################
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
```

np.ones的使用！！！！！！！！！！

```
D = np.ones((m,1))/m      ## 这样的话，就是 m,1 维的
D = np.ones((m))/m        ## 这样的话就是 m,  ！！！！！
```

在用数学工具的时候，不用要python的math包，改成用np.ex(pnp.array)

**下面是用Python实现的经典的quicksort算法例子：**

  ```python
def quicksort(arr):
 2    if len(arr) <= 1:
 3        return arr
 4    pivot = arr[len(arr) / 2]
 5    left = [x for x in arr if x < pivot]
 6    middle = [x for x in arr if x == pivot]
 7    right = [x for x in arr if x > pivot]
 8    return quicksort(left) + middle + quicksort(right)
 9print quicksort([3,6,8,10,1,2,1])
10# Prints "[1, 1, 2, 3, 6, 8, 10]"
  ```

**列表推导还可以包含条件：**

```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares  # Prints "[0, 4, 16]"
```

**迭代获取字典内容**

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for key,value in d.items():
    print(key,value)
#person 2
#cat 4
#spider 8
```

**字典推导Dictionary comprehensions和列表推导类似，但是允许你方便地构建字典。**

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square  # Prints "{0: 0, 2: 4, 4: 16}"
```

**SciPy提供了一些操作图像的基本函数。**

### [Pyhton包argparse的使用](http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html)