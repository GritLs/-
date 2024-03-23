# 习题 
## 1.1.  

**解答**  
**1. 伯努利模型**
伯努利模型可写为：
 $$P_p(X=x)=p^x(1-p)^{1-x},\ 0\le p \le 1$$
 伯努利模型的假设空间为：  
 $$\mathcal{F}=\{P|P_p(X)=p^x(1-p)^{1-x},p\in[0,1]\}$$  
 其中$x$的取值为$0$或$1$。  
 
 **2.伯努利模型的极大似然估计的统计学习方法三要素**  
 模型:伯努利模型  
 策略:经验风险最小化。  
 算法:极大化似然 $\arg \underset p \max P(X|p)=\arg \underset p \max L(p|X)$ ,其中$L(p|X)$是似然函数,由于伯努利模型仅有一个参数$p$,所以$\theta = p$.
 >推导: $L(p|X)=f(x_1;p)f(x_2;p)\dots f(x_n;p)$   
 $\log L(p|X)=\log f(x_1;p)+\dots+\log f(x_n;p)$  
 $\log L(p|X) = k\log p+(n-k)\log(1-p)$  
 $\frac{\partial \log L(p|X)}{\partial p}=\frac{k}{p}-\frac{(n-k)}{1-p}=0$  
 $p=\frac{k}{n}$  

 **3.伯努利模型的贝叶斯估计的统计学习方法三要素**  
模型:伯努利模型  
策略:结构风险最小化  
>关于结构风险最小化的推导:  
$\arg \underset{p}{\max}\pi(p|X)=\arg \underset{p}{\max}\frac{P(X|p)\pi(p)}{\int P(X|p)\pi(p)dp}=\arg \underset p \max P(X|p)\pi(p)$  
等价于:  
$\arg \underset p \max( \log P(X|p)+\log \pi(p))$  
其中$\log \pi(p)$先验分布可解释为结构风险中的正则化项。


算法:最大化后验概率:$\arg \underset p \max\ \pi(p|X)$，如果我们取$\pi(x)$为均匀分布，则最大化后面分布等价于极大似然估计。

## 1.2. 
**1.经验风险最小化定义**  

假设有数据集$D=\{(x_1,y_1),\dots,(x_n,y_n)\}$，我们要在假设空间$\mathcal{{F}}$中找到一个模型$f$，满足
$$\underset{f\in \mathcal{F}}{\min}\frac{1}{N}\sum_{D}L(y_i,f(x_i))$$  

**2.对数损失函数**  

$$L(Y,P(Y|X))=-\log P(Y|X)$$  

**3.推导**  
$$
\underset{f\in \mathcal{F}}{\min}\frac{1}{N}\sum_{D}L(y_i,f(x_i))=\\\underset{f\in \mathcal{F}}{\min}\frac{1}{N}\sum_{D}-\log P(Y|X)=\\\frac{1}{N}\underset{f\in \mathcal{F}}{\max}\sum_{D}\log P(Y|X)=\\
\frac{1}{N}\underset{f\in \mathcal{F}}{\max}\log \prod_{D}P(Y|X) \tag{1}
$$ 
似然函数的定义为:
$$
L(\theta|X)=\prod_{D}P_{\theta}(Y|X) \tag{2}
$$ 
$\log$不改变单调性，故最大化(2)等价于(1)，故得证。