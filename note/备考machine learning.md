**图片上的每句话都要看一下！！！！！！！！！！！！！！！！！**

![image-20181022175750072](/Users/yunqingqi/Desktop/note/image-20181022175750072.png)

![image-20181106190244908](/Users/yunqingqi/Desktop/note/image-20181106190244908.png)

![image-20181106190645087](/Users/yunqingqi/Desktop/note/image-20181106190645087.png)

![image-20181106190814825](/Users/yunqingqi/Desktop/note/image-20181106190814825.png)

![image-20181106191033199](/Users/yunqingqi/Desktop/note/image-20181106191033199.png)

![image-20181106191145961](/Users/yunqingqi/Desktop/note/image-20181106191145961.png)

红色区域里出现的蓝色点就是noise！！！

![image-20181106191614575](/Users/yunqingqi/Desktop/note/image-20181106191614575.png)

![image-20181106191858439](/Users/yunqingqi/Desktop/note/image-20181106191858439.png)

![image-20181106192120793](/Users/yunqingqi/Desktop/note/image-20181106192120793.png)

 ![image-20181106192434784](/Users/yunqingqi/Desktop/note/image-20181106192434784.png)

![image-20181106192637597](/Users/yunqingqi/Desktop/note/image-20181106192637597.png)

![image-20181106192736968](/Users/yunqingqi/Desktop/note/image-20181106192736968.png)

这上面就写了 model selection 的好处

![image-20181106193026672](/Users/yunqingqi/Desktop/note/image-20181106193026672.png)

![image-20181106193413141](/Users/yunqingqi/Desktop/note/image-20181106193413141.png)

![image-20181106193848804](/Users/yunqingqi/Desktop/note/image-20181106193848804.png)

**每次留一个subset for validation , 剩下的是training～  k 一般取5就可以了，k太大会浪费时间**

![image-20181106194534314](/Users/yunqingqi/Desktop/note/image-20181106194534314.png)

**归一化我们的 feature 数据**

![image-20181106194722111](/Users/yunqingqi/Desktop/note/image-20181106194722111.png)

下面这种方法是老师讲的

![image-20181107131202980](/Users/yunqingqi/Desktop/note/image-20181107131202980.png)

但是我觉得还是另外一个笔记里那个老师讲的更好一点（从loss function下手直接推）

**去看另一个笔记里的svm讲解**

![image-20181107135414762](/Users/yunqingqi/Desktop/note/image-20181107135414762.png)

![image-20181107140420472](/Users/yunqingqi/Desktop/note/image-20181107140420472.png)

![image-20181107140429992](/Users/yunqingqi/Desktop/note/image-20181107140429992.png)

![image-20181107141912541](/Users/yunqingqi/Desktop/note/image-20181107141912541.png)

![image-20181107142440032](/Users/yunqingqi/Desktop/note/image-20181107142440032.png)

我们的目标函数 J 后面加了个 ![image-20181107142849158](/Users/yunqingqi/Desktop/note/image-20181107142849158.png)，就是说我们想让没分对的点少一点。

下面我们 写成![image-20181107143122399](/Users/yunqingqi/Desktop/note/image-20181107143122399.png)？？？？？？？？

![image-20181107142934470](/Users/yunqingqi/Desktop/note/image-20181107142934470.png)

#### **log is a convex function**

GMM 得把ppt 全打印下来，没太看懂

![image-20181107192742338](/Users/yunqingqi/Desktop/note/image-20181107192742338.png)

Adaboost 里，我们的错误率不能高于0.5，如果高于0.5，那么$\alpha$ < 0, 那么![image-20181107192831712](/Users/yunqingqi/Desktop/note/image-20181107192831712.png) 就会有减号，也就是说该 weak classifier(this prediction score) doesnt contribute to our final ensemble classifier, it should be moved 

**还有一个需要注意！！！！！！！！！：我们的分类器明明是线性组合，为什么final classifier的图形不是线性的？** ： 这是因为  $h_t(x)$ has a sign 导致的，看代码 ：

![image-20181107195001332](/Users/yunqingqi/Desktop/note/image-20181107195001332.png)

我们最后返回的 s 的值 就是 $h_t(x)$ 

![image-20181108130607070](/Users/yunqingqi/Desktop/note/image-20181108130607070.png)

从下面可以看到，regression 和 classification 前两个步骤一样，只是最后一个(learning part)不一样

![image-20181108130817621](/Users/yunqingqi/Desktop/note/image-20181108130817621.png)

![image-20181108131241026](/Users/yunqingqi/Desktop/note/image-20181108131241026.png)

我们计算的是 reconstruction error ？？，和 PCA 一样（感觉老师讲解的pca还是要回去看一下推导～～～）

PCA也有可能失败，设想另一个极端，如果数据中噪声非常大，而有用信息非常微弱，则这里大variance的维度是噪声的维度，降维时反而是有用信息被去掉了。所幸的是，我们所遇到的绝大部分数据中噪声都是相对小的，要不然分析数据也就没有希望了。

![image-20181108132048188](/Users/yunqingqi/Desktop/note/image-20181108132048188.png)

那么svm里为什么要对 $\varepsilon $ ['epsɪlɒn] 做 regularization ？？？？ 老师说 svm 用的是L2 norm, 也就是说$\frac{1}{2}||w||^2$ 是我们的 正则化，而 $\varepsilon $ 不是～！！！ (这也就解释了下面这个图的正确)

![image-20181108135302542](/Users/yunqingqi/Desktop/note/image-20181108135302542.png)

**只是说我们的第一项![image-20181108140124919](/Users/yunqingqi/Desktop/note/image-20181108140124919.png) 最后可以转化成一个constrain ![image-20181108140137259](/Users/yunqingqi/Desktop/note/image-20181108140137259.png) ，注意我们的优化目标（损失函数）就只有 $\frac{1}{2}||w||^2$ 了而已。。。。。**

![image-20181108133751363](/Users/yunqingqi/Desktop/note/image-20181108133751363.png)

SVR 和 LR 的区别是，for LR, only the data points that are exactly on the line **will not** incur a loss, al the other data points that are off line will incur a loss. For SVR,  it extends the straight line into a margin, if the data points in the margin, it **will not** incur a loss.

**SVR 和 SVM 的损失函数其实不一样的，SVR 的损失是就是 $|y-f(x)| \leq \epsilon $  也就是  $y-f(x) \leq \epsilon $ 和 $f(x)-y \leq \epsilon $**

![image-20181108141357453](/Users/yunqingqi/Desktop/note/image-20181108141357453.png)

**在上面就已经说到了SVR 和 SVM 的损失函数其实不一样的，看看上面右图的横坐标，是y - f(x,w),和 SVM 的yf(x)不一样哦！！！！** 

**for soft margin，我们有 2 个 slack veriables !!!!!!! 两边各一个！！！！！！！！！！！！ ** 

![image-20181108141644527](/Users/yunqingqi/Desktop/note/image-20181108141644527.png)

![image-20181108142900879](/Users/yunqingqi/Desktop/note/image-20181108142900879.png)

**SVM 和 SVR 的推导，看桌面上下载的notes ！！！！推导一遍！！！！**

**L1 比 L2 更robost**：

![image-20181108144138379](/Users/yunqingqi/Desktop/note/image-20181108144138379.png)

**在笔记 machine 再来一遍里也有说明！！！！**

![image-20181108144341804](/Users/yunqingqi/Desktop/note/image-20181108144341804.png)

对于polunomial kernels 里的 d，我们一般让它为 2，in practice, RBF 效果比 polynomial 的要好

能应用kernel的关键条件：

![image-20181107131830953](/Users/yunqingqi/Desktop/note/image-20181107131830953.png)

我们的原始式子里有 $X^TX$ 或者 $\sum x_ix_i^T$（inner product）

![image-20181108144621770](/Users/yunqingqi/Desktop/note/image-20181108144621770.png)

