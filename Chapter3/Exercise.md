# 习题
## 4.1
证明（4.8）：$P(Y=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N},\ \ k=1,2,\cdots,K$  
假设$P(Y=c_k)$的概率为$p$，其中$c_k$在随机变量$Y$中出现的次数为$m=\sum_{i=1}^{N}I(y_i=c_k)$，似然函数为：
$$
\begin{align}
L(p|Y)&=f(Y|p)\\
&=C_N^mp^m(1-p)^{N-m}
\end{align}
$$
则有:
$$
\begin{align}
\log L(p|Y)&=\log C^m_N + m\log p +(N-m)\log(1-p)\\
\frac{\partial \log L(p|Y)}{\partial p}&=\frac{m}{p}-\frac{N-m}{1-p}=\frac{m-Np}{p(1-p)}=0\\
p&=\frac{m}{N}=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}
\end{align}
$$
得证。  

证明（4.9）：$P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^{N}I(x^{(j)}_i=a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}\\ 其中 j=1,2,\cdots,n;\ l=1,2,\cdots,S_j;\ k=1,2,\cdots,K$  
在朴素贝叶斯法做了条件独立性的假设。具体来说：
$$
P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}|Y=c_k)\\=\prod_{j=1}^{n}P(X^{(j)}=x^{(j)}|Y=c_k)
$$
假设$P(X^{(j)}=a_{jl}|Y=c_k)$的概率为$p$，其中$c_k$在随机变量$Y$中出现的次数$m=\sum_{i=1}^{N}I(y_i=c_k)$，$y=c_k$和$x^{(j)}=a_{jl}$同时出现的次数$q=\sum_{i=1}^{N}I(x^{(j)}_i=a_{jl},y_i=c_k)$，似然函数:
$$
L(p|X,Y)=f(X,Y|p)=C^q_mp^q(1-p)^{m-q}
$$
同理，最大化$L(p|X,Y)$ 可得$p=\frac{q}{m}$.

## 4.2
**贝叶斯估计的一般步骤**  
1. 确定参数$\theta$的先验概率$p(\theta)$
2. 根据样本集$D=x_1,x_2,\cdots,x_n$，计算似然函数$P(D|\theta)=\prod_{i=1}^{n}P(x_n|\theta)$ 
3. 利用贝叶斯公式，求$\theta$的后验概率$P(\theta|D)=\frac{P(D|\theta)P(\theta)}{\int P(D|\theta)P(\theta)d\theta}$
4. 计算后验概率分布参数$\theta$的期望$\hat \theta = \int \theta P(\theta|D)d\theta$

证明（4.11）：
$$
P_\lambda(Y=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)  \lambda}{N+K\lambda},\ \ k=1,2,\cdots,K
$$
假设$P_\lambda(Y=c_k)=u_k$服从参数为$\lambda$的Dirichlet分布；随机变量$Y$出现$y=c_k$的次数为$m_k$  
Dirichlet分布是Beta分布的推广，适用于多元的情况。
以下为Dirichlet分布的概率密度函数：
$$
f(x_1,\cdots,x_K;\alpha_1,\cdots,\alpha_K)=\frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)}\prod_{k=1}^{K}x_k^{\alpha_k-1}\\
s.t. \sum_{k=1}^{K}x_k=1,\ x_k\ge0,\ \alpha_j>0,\ \alpha_0=\sum_{k=1}^{K}\alpha_k
$$
$$
似然：p(D|u)=p(x_1,\cdots,x_n|u)=\prod_{k=1}^{K}u_k^{m_k}
$$
<!-- 关于$\{m_i\}$的联合概率分布 (多项概率分布)：
$$
p(m_1,\cdots,m_K|u,N)=\frac{N!}{m_1!\cdots m_K!}\prod_{k=1}^{K}u_k^{m_k}
$$ -->
先验：
$$
p(u|\alpha)=\frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)}\prod_{k=1}^{K}u_k^{\alpha_k-1}\\
$$
后验：
$$
p(u|D,\alpha)\propto p(D|u)p(u|\alpha)\propto \prod_{k=1}^{K}u_k^{\alpha_k+m_k-1}
$$
可以得出，后验仍然满足Dirichlet分布的形式。具体来说(TODO，这里的系数是怎么直接得出来的？):
$$
p(u|D,\alpha)=Dir(u|\mathbf{\alpha}+\mathbf{m})=\frac{\Gamma(\alpha_0+N)}{\Gamma (\alpha_1+m_1)\cdots \Gamma (\alpha_K+m_K)}\prod_{k=1}^{K}u_k^{\alpha_k+m_k-1}
$$

在我们的例子中所有的$\alpha_k=\lambda$，而一般形式的$Dir(u|\alpha)$的期望$E(u_k)=\frac{\alpha_k}{\sum_{k=1}^{K}\alpha_k}$。于是在我们的例子中
$$
E(u_k)=\frac{\lambda+m_k}{\sum_{k=1}^{K}\lambda+m_k}=\frac{\lambda+m_k}{K\lambda+N}=\frac{\lambda+\sum_{i=1}^{N}I(y_i=c_k)}{K\lambda+N}
$$
得证。  

证明公式4.10：
条件假设$P_\lambda(X^{j}=a_{jl}|Y=c_k)=u_l$，其中$l=1,2,\cdots,S_j$；出现$x^{(j)}=a_{jl},y=c_k$的次数为$m_l$。   
  
先验概率: 
$$
P(u)=\frac{\Gamma(\lambda S_j)}{\Gamma(\lambda)^{S_j}}\prod_{l=1}^{S_j}u_l^{\lambda-1}
$$

似然函数:
$$
P(m|u)=u_1^{m_1}\cdots u_{S_j}^{m_{S_j}}=\prod_{l=1}^{S_j}u_l^{m_l}
$$
根据贝叶斯公式$P(u|m)=\frac{P(m|u)P(u)}{p(m)}$有：
$$
P(u|m,\lambda)\propto p(m|u)p(u|\lambda)\propto \prod_{l=1}^{S_j}u_l^{\lambda+m_l-1}
$$
与4.11的证明同理
$$
\begin{align}
E(u_l)&=\frac{\alpha_l}{\sum_{l=1}^{S_j}\alpha_l}\\
&=\frac{\lambda+m_l}{\sum_{l=1}^{S_j}(\lambda+m_l)}\\
&=\frac{\lambda+m_l}{S_j\lambda+\sum_{l=1}^{S_j}m_l}\\
&=\frac{\sum_{N}^{i=1}I(x_i^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^{N}I(y_i=c_k)+S_j\lambda}
\end{align}
$$
得证。

[参考1 P75-77](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf?ranMID=46131&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-vf7mtWsNSRUxgdcBLBDKJQ&epi=a1LgFw09t88-vf7mtWsNSRUxgdcBLBDKJQ&irgwc=1&OCID=AIDcmm549zy227_aff_7806_1243925&tduid=%28ir__ztqaiaf6f9kfbk3o1933exvyvv2xd2cqboj693zv00%29%287806%29%281243925%29%28a1LgFw09t88-vf7mtWsNSRUxgdcBLBDKJQ%29%28%29&irclickid=_ztqaiaf6f9kfbk3o1933exvyvv2xd2cqboj693zv00)  
[参考2](https://datawhalechina.github.io/statistical-learning-method-solutions-manual/#/chapter04/ch04)