# Normalization Effects on Neural Networks

Authors of this repository are Konstantinos Spiliopoulos and Jiahui Yu.

This repository contains code supporting the article

Konstantinos Spiliopoulos and Jiahui Yu, "Normalization effects on shallow neural networks and related asymptotic expansions", 2021, AIMS Journal on Foundations of Data Science, June 2021, Vol. 3, Issue 2, pp. 151-200.

Journal version: https://www.aimsciences.org/article/doi/10.3934/fods.2021013?viewType=html

ArXiv preprint: https://arxiv.org/abs/2011.10487.

First read the Read Me file to run the code.

To report bugs encountered in running the code, please contact Konstantinos Spiliopoulos at kspiliop@bu.edu or Jiahui Yu at jyu32@bu.edu

# Short exposition

Cosnider for example the one layer neural network

$$
\begin{align}
g^N(x;\theta) &= \frac{1}{N^{\gamma}} \sum_{i=1}^{N} C^i\sigma(W^i x+b^i)
\end{align}
$$

where $\gamma\in[1/2,1]$ and $\theta = (C^1,\ldots, C^N, W^1, \ldots, W^N, b^1,\dots, b^N) \in \mathbb{R}^{(2+d)N}$ are the parameters to be learned. Here the two extreme cases are $\gamma=1/2$ that
corresponds to the popular Xavier initialization and $\gamma=1$ corresponds to the mean-field normalization. 

Let the objective function be for example

$$
\begin{align}
\mathcal{L}(\theta) &= \frac{1}{2} E_{X,Y} (Y-g^N(X;\theta))^2
\end{align}
$$

We learn the parameters $\theta$ using stochastic gradient descent. The choice of the learning rate is theoretically linked to the $\gamma$ parameter. In particular, the theory developed in the paper above suggests that for this case the learning rate should be chosen to be of the order of

$$
\alpha_{N,\gamma}=\frac{\alpha}{N^{2(1-\gamma)}} 
$$

for $\alpha$ a constant of order one.

The goal of the aforementioned paper is to study  the performance of neural networks scaled by $1/N^{\gamma}$ with $\gamma\in [1/2, 1]$. 

The theoretical results derive an asymptotic expansion of the neural network's output with respect to $N$. This expansion demonstrates the effect of the choice of  $\gamma$ on bias and vairance. In particular, for large and fixed $N$, the variance goes down monotonically as $\gamma$ increases to $1$.

The numerical results of the paper, done on MNIST and CIFAR10 datasets, demonstrate that train and test accuracy monotonically increase as $\gamma$ goes to 1. 

The conclusion is that the mean-field normalization $\gamma=1$ is clearly the optimal choice!! But, for this to be relaized the learning rate has to be chosen by the theoretically informed choice stated above.

See an example here on the MNIST data set with cross-entropy loss function with $N=3000$ hidden units (taken from the accompanying paper stated above). 

![plot_mnist_ce_h3000_e500_b20_test](https://user-images.githubusercontent.com/106413949/172763587-1c41126e-368a-4f5f-8ab1-5c1b917dcc23.png)
