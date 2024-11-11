# Adversarial Robustness through Random Weight Sampling

近年的研究已经证实了噪声对于对抗攻击的重要作用。但是防御的效果和加噪的参数高度绑定，常常需要进行手动调整。

因此本文提出将随机化的权重添加到优化过程中以充分利用防御机制的潜能。为了更好的利用随即参数，本文从理论上分析了随即参数对梯度近似以及自然性能之间的关系。

本文通过考虑预测偏差和梯度相似性，来对随即权重划定上下界。提出了CTRW（constrained trainable random weight），可以在optimization过程中添加随机的weight参数，并且用上下界对其进行约束来进行acc与鲁棒性之间的折中。



## 1. Introduction

目前现有的随机防御算法总是将添加的噪声的均值和方差设为一种超参数，并通过大量的实验经验选取。这种方式不能很好的选择超参数，从而影响鲁棒性。本文通过理论指导，选择噪声的参数而不是通过手动调整。具体来说，本文通过随机噪声对梯度以及自然表现之间的关系来进行理论指导。



进一步，本文提出了CTRW方法，可以与神经网络并行训练噪声的相关参数，并且在每次更新后能够根据建立的约束条件对方差进行约束，保证参数始终在最优范围内。



## 2. Randomized parameter optimization with constraints

在引入随机噪声之后，模型变成了一个模型组H，在训练阶段和推理阶段会发生变化，分别用$h_1,h_2$表示。在这种情况下，即使是白盒攻击也可以视为与黑盒攻击类似。

![image-20241107142021988](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241107142021988.png)

假设r为施加了随即权重的层，$W^r$满足$N(\mu,\sigma^2)$。为保证随机权重能够与训练数据相适应，在训练阶段就引入随机化噪声。但随机化的噪声层不可约，需要通过重新参数化方式进行优化
$$
W^r=\mu+\delta.\sigma
$$
其中$\delta$是遵循$N(0,1)$的随机量。对于h1和h2来说，仅仅是该数值有区别。

为保证随机化对鲁棒性的有效性，需保证$\sigma$比较大。但是大的话会影响系统的自然表现。因此为了统一优化$\sigma$本文提出了一种对其上下界的约束，来同时保证系统的自然性能和鲁棒性。用$W[i]$表示$h_i$的权重，目标函数可以写为
$$
H^*=argmin_{h_{W[2]}}\ E_{x,y}[L(h_{W[2]}(\hat x),y)]\\
\hat x=argmax\ L(h_{W[1]}(\hat x),y)\\
\sum\sigma\in[A,B]
$$
其中A，B为随机化权重的上下界。



### 2.1 lower bound for gradient similarity

为保证随机防御的成功，需要保证在随机化前后的梯度产生足够大的差异。在此使用余弦相似度来衡量之间的距离。
$$
cos(\nabla(x,W[1]),\nabla(x,W[2]))
$$
其中随机权重通过神经网络中各层输入的**逐元素乘法（Hadamard product）**产生。为保证梯度差异足够大，需要对cos设定一个上限。

但是由于随机权重中$\delta$是一个无界的高斯分布，所以我们并不能保证每一次cos相似度都满足整个条件，只能提供一个概率界。
$$
\mu,\sigma=argmax\ P(cos(\nabla(x,W[1]),\nabla(x,W[2]))<\epsilon_{\nabla})
$$
当模型参数固定时，对于第0层的梯度和第r-1层的梯度，两者之间是存在一个常数的矩阵的。因此对于1和2来说其都是一致的，可以不去考虑。而对于r以及后面的层上，可以表示为
$$
cos(\nabla(x,W[1]),\nabla(x,W[2]))=\\cos(\nabla^{r,r+1}[1],\nabla^{r,r+1}[2])cos(\nabla^{r+1,n}[1],\nabla^{r+1,n}[2])-sin(\nabla^{r,r+1}[1],\nabla^{r,r+1}[2])sin(\nabla^{r+1,n}[1],\nabla^{r+1,n}[2])
$$
且由于我们需要保证y[1]和y[2]之间要尽可能地相似，而$\nabla^{r+1,n}[1],\nabla^{r+1,n}[2]$仅仅与y[1]和y[2]之间是乘法关系，因此$\nabla^{r+1,n}[1],\nabla^{r+1,n}[2]$之间就要尽可能地相似。

因此上述的cos距离，其实主要由$cos(\nabla^{r,r+1}[1],\nabla^{r,r+1}[2])$这一项决定。而随机层又是输入与权重之积，所以该层的导数就恰好是随机权重。
$$
\mu,\sigma=argmax\ P(cos(W[1],W[2])<\epsilon_{r})
$$
将公式完全展开可以得到

![image-20241108212716928](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241108212716928.png)

需要注意的是$\delta$是一个标准正态分布，其他变量都是常数，因此对于其可以用cumulative的高斯分布来对该概率进行限制。

计算可以得到

![image-20241108212915503](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241108212915503.png)

在此基础上，定义与权重相关的$\sigma$的上界。



引理1：如果一个固定的$W[1]$有m个元素，约束常数$\epsilon_r$，$\alpha<0$。可以计算$\epsilon'_r=\epsilon_r\times ||W[1]||\ ||W[2]||$。假设$\sum_i^m\mu_i>>0$。那么对于cos距离的限制满足以下的结果：
$$
if\ \sum_i^m\sigma_i>\frac{\epsilon_r'-\sum_i^m\mu_i}{\alpha\sum_i^m|W[1]_i|}>0\\
P(cos(W[1],W[2])<\epsilon_{r})>F(\alpha)
$$
因此在训练过程中，每次优化后都将检查$\sigma$，当其小于下界后通过加上平均偏差进行回调。





### 2.2 upper bound for natural performance

![image-20241108105207293](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241108105207293.png)

上界用于控制clean的acc精度。即控制$|y_1-y_2|$足够小。

由于卷积和normalization都是线性操作，所以可以在进行Hadamard乘积之前将每一层的随机权重的变化表示出来，如图所示。

在经过卷积之后，每个随机权重变成了多个独立同分布的随机变量之和，结果也是一个高斯分布。

我们需要的是

![image-20241108213336429](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241108213336429.png)

将其展开可以得到

![image-20241108213350325](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241108213350325.png)

对于线性部分，有$H^{i,j}(W^r\odot x)=H^{i,j}(W^r)\odot H^{i,j}(x)$

非线性部分，$ReLU(a\odot b)>=ReLU(a)\odot ReLU(b)$

所以可以近似的获得$H^{i,j}(W^r\odot x)=H^{i,j}(W^r)\odot H^{i,j}(x)$

![image-20241110150820888](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110150820888.png)

除去最后一层的全连接层，对于$H^{r+1,n-1}(W[1])$，由于高斯分布的求和也是高斯分布，因此其可以表示为$N(\mu',\hat\sigma)=\mu'+\delta\odot\hat\sigma$。所以有

![image-20241110153744552](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110153744552.png)

可以看到$\delta[1]-\delta[2]=N(0,2)$。所以$P(\delta[1]-\delta[2]<\beta)=F(\frac{\sqrt 2\beta}{2})$

所以如果我们希望获得

![image-20241110154433477](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110154433477.png)

经过计算后需要满足

![image-20241110155223299](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110155223299.png)

其中$\sigma'$可以通过后续层中的参数计算获得。



### 2.3 adversarial training and defense

![image-20241110171817840](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110171817840.png)



对于adversarial training，采用类似黑盒的方法，在训练阶段也对噪声的参数$\mu,\sigma$进行更新，但是每次更新都要在上述的限定范围里。但是在实际的过程中，他选定的范围是固定的，也能达到效果。

在adversarial defense阶段，只需要将生成对抗样本和进行推理的噪声进行随机采样即可。



## 4. experiments

在CIFAR和ImageNet上进行测试。

实验里$\alpha$选了个大于0的值，考虑到OpenReview中的讨论，这里应该是修改过，但是最后的实验结果没有改。



### 4.2 results and discussion

![image-20241110173158698](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110173158698.png)

分别设置了其他两组对比试验。without constrain表示在训练过程中不设置限制条件，这一组实现的效果更差，而且对大多数情况下， $\sigma$都趋向于0，发生收敛的情况。另一组fixed std，保持最小值而不进行训练。



![image-20241110174730271](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110174730271.png)

为进一步说明有无边界的区别，1234为无边界的CIFAR10、CIFAR100上的ResNet18和WRN34。可以看出有边界的训练可以将$\sigma$收敛到更高的值上，并且收敛的效果更好，所以也能在鲁棒性上取得更好的结果。



![image-20241110211203761](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110211203761.png)

与其他随机化防御方法的对比



### 4.3 ablation study

下界的有效性：$\alpha$被选定为小于0，为了验证这个限制的有效性，本文分别计算了在加入约束和不加入约束时满足$cos(W^r[1],W^r[2])<\epsilon_r$的样本数量。满足该条件的样本被称为PS，结果如下所示。可以发现在没有约束的情况下满足该条件的样本数为0，在有约束条件下，满足该条件的样本数更多。因此可以得知，下界确实是有效的。

![image-20241110211834328](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110211834328.png)



在强PGD下的鲁棒性：对比不同强度PGD下的CTRW的有效性，可以看到

![image-20241110211948426](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110211948426.png)



CTRW层的位置：由于上界和下界和所在的层数是绑定的，本文验证了在不同层的结果。对比了不同层的区别以及在卷积层第一层的不同神经元位置的区别。在下图中，position表CTRW的位置，0表示第一个卷积层，5表示FC。可以观察到CTRW在前几层是更有效果的。所以最合适的层是在第一个卷积层。进一步，在层内部的位置需要进一步讨论。可以在输入的位置、输出的位置、输入和输出的位置、层内的filter添加噪声。结果如下图所示

![image-20241110212534748](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110212534748.png)

![image-20241110212558985](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110212558985.png)



预测结果的随机性：为了让添加的随机噪声不影响clean结果的准确性，本文测试了结果的变化幅度，发现很小。

![image-20241110212842130](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110212842130.png)



最后测试了上下界的重要性，发现上下界都是必须的。

![image-20241110212910030](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20241110212910030.png)
