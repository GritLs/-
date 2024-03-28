# 朴素贝叶斯法
### 基本过程
**强假设：样本的各个特征之间相互独立**  
$$
P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\dotsb,X^{(n)}=x^{(n)}|Y=c_k)\\
=\prod_{j}P(X^{j}=x^{j}|Y=c_k)
$$

其中$c_k$为label，$X^{(j)}$为样本的第$j$个特征。

$$
y=f(x)=\arg \underset{c_k}{\max} \frac{P(Y=c_k)\prod_{j} P(X^{j}=x^{j}|Y=c_k)}{\sum_k P(Y=c_k)\prod_j P(X^{j}=x^{j}|Y=c_k)} \tag{1}
$$

其中 $\sum_k P(Y=c_k)\prod_j P(X^{j}=x^{j}|Y=c_k)$为定值，故(1)等价于
$$
y=f(x)=\arg \underset{c_k}{\max}\ P(Y=c_k)\prod_{j} P(X^{j}=x^{j}|Y=c_k) 
$$

因此，我们需要学习的参数为$P(Y=c_k)$ 和 $P(X^j=x^j|Y=c_k)$。具体来说，对于$P(X^j=x^j|Y=c_k)$，我们需要学习的是
$$
P(X^j=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^{N} I(x_i^j=a_{jl},y_i=c_k)}{\sum_{i=1}^{N} I(y_i=c_k)}
$$
其中$I$为判别函数，用于判断该实例是否存在于样本集中，$a_{jl}$表示第$j$个特征的第$l$种情况。  

预测过程中，输入一个样本，我们通过查表获得各个标签的概率$P(c_k)$和在各个标签条件下的特征值的条件概率$P(X^j=x^j|c_k)$来计算后验概率$P(c_k|X)$，通过选择最大的后验概率来预测标签。这种方法的好处是实现简单，缺点是假设**样本的各个特征之间相互独立**的前提条件过强，很多情况下不适用。

## 扩展：贝叶斯网络
将Markov链进行扩展，每个随机变量$x_i$仅与$x_{\pi(i)}$有关，于是联合概率分布可以写作：
$$
P(x_1,x_2,\cdots,x_n)=\prod_{i=1}^{n}P(x_i|x_1,\cdots,x_{i-1})\approx \prod_{i=1}^{n} P(x_i|x_{i-k+1},\cdots,x_{i-1})\\= \prod_{i=1}^{n} P(x_i|x_{\pi(i)})
$$

基本关系：
- parent->ancestor
- child->descendant
- co-parent
- Markov blanket ：parent, child, co-parent

定理：
- 给定一个节点$x_i$的Markov blanket则有：
$$
P(x_i|MB(x_i),Y)=P(x_i|MB(x_i))
$$
- 给定一个节点$x_i$的父节点$A(x_i)$，则$x_i$相对于它的nondescedants节点$ND(x_i)$条件无关:
$$
P(x_i|A(x_i),ND(x_i))=P(x_i|A(x_i))
$$

## 训练
**三大类训练贝叶斯模型的方法**：
- Maximum likelihood estimation（MLE）
$$
\Theta^{MLE}=\arg \underset{\Theta}{\max} P(D|\Theta)
$$ 
- Maximum a posteriorp (MAP) (引入先验分布)
$$
\Theta^{MAP} = \arg \underset \Theta \max P(\Theta)P(D|\Theta)
$$
- Bayesian estimation （计算复杂度最高的情况，需要计算$\Theta$在整个分布下的积分）
$$
\text{Find}\ P(\Theta|D)
$$
### 1. Maximum likelihood estimation（MLE）
**无隐变量的情况下**  
$$
D=\{X_i\}|_{i=1}^{N}\ \ \ \ \ \ \ \ X_i=x_1^i,x_2^i,\cdots,x_K^i \\
\Theta = \{\theta_k\}|_{k=1}^K\ \ \ \ \ \ \ \ \theta_k=P(x_k|x_{\pi(k)})\\
\Theta^{MLE}=\arg \underset{\Theta}{\max}\sum_{i=1}^{N}\log P(x^i_1,x^i_2,\cdots,x^i_K|\Theta)\\
=\arg \underset{\Theta}{\max}\sum_{i=1}^{N}\sum_{k=1}^{K}\log P(x^i_k|x^i_{\pi(k)},\theta_k)
$$
计算方法（相对频率）：
$$
\theta_k^{MLE}(x_k=v|x_{\pi(k)}=v_{\pi})=\frac{I(x_k=v,x_{\pi(k)}=v_{\pi})}{I(x_{\pi(k)}=v_{\pi})}
$$

**有隐变量的情况 (TODO)**

### 2. Maximum a posteriorp (MAP)
$$
\begin{align}
\Theta^{MAP} &= \arg \underset{\Theta}{\max}\log(P(\Theta)P(D|\Theta))\\
&=\arg \underset{\Theta}{\max}(\log P(\Theta)+\log P(D|\Theta))\\
&=\arg \underset{\Theta}{\max}(\log P(\Theta)+\sum_{i=1}^{N}\sum_{k=1}^{K}\log P(x^i_k|x^i_{\pi(k)},\theta_k))
\end{align}
$$

### 3. Bayesian Estimation（一般要引入共轭分布？）
后验概率 $P(\Theta|D)=\frac{P(D|\Theta)P(\Theta)}{P(D)}=\frac{P(D|\Theta)P(\Theta)}{\int P(D|\Theta)P(\Theta)d\Theta}$  
输出: $P(y|x,D)=\int_{\Theta}P(y,\Theta|x,D)d \Theta=\int_{\Theta}P(y|x,\Theta)P(\Theta|D)d\Theta$

### 总结
Maximum Likelihood Estimation (MLE)可以理解为经验风险最小化，Maximum a Posteriorp(MAP)通过引入先验分布可以理解为结构风险最小化。与Bayesian Estimation相比，这两种方式都有直接确定的$\Theta$参数值，在这一给定参数值的情况下进行预测。而Bayesian Estimation可以理解为给出了$\Theta$的后验分布$P(\Theta|D)$，预测的结果是$\Theta$在这一后验分布的期望。

[参考](https://www.bilibili.com/video/BV1PA411f7z4/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=2e4c0197696e9aefa1f2f8309577223d)