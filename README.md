# Normalization Effects on Neural Networks (NENN)

Authors of this repository are Konstantinos Spiliopoulos and Jiahui Yu.

This repository contains code supporting the article

Konstantinos Spiliopoulos and Jiahui Yu, "Normalization effects on shallow neural networks and related asymptotic expansions", 2021, AIMS Journal on Foundations of Data Science, June 2021, Vol. 3, Issue 2, pp. 151-200.

Journal version: https://www.aimsciences.org/article/doi/10.3934/fods.2021013?viewType=html

ArXiv preprint: https://arxiv.org/abs/2011.10487.

First read the Read Me file to run the code.

To report bugs encountered in running the code, please contact Konstantinos Spiliopoulos at kspiliop@bu.edu or Jiahui Yu at jyu32@bu.edu

# Short exposition--Achieving good accuracy with less need for fine tuning. 

Cosnider for example the one layer neural network

$$
\begin{align}
g^N(x;\theta) &= \frac{1}{N^{\gamma}} \sum_{i=1}^{N} C^i\sigma(W^i x+b^i)
\end{align}
$$

where $\gamma\in[1/2,1]$ and $\theta = (C^1,\ldots, C^N, W^1, \ldots, W^N, b^1,\dots, b^N) \in \mathbb{R}^{(2+d)N}$ are the parameters to be learned. Here the two extreme cases are $\gamma=1/2$ that
corresponds to the popular Xavier initialization, see [2], and $\gamma=1$ corresponds to the mean-field normalization, see [1,3,4,5,6,7]. 

Let the objective function be for example

$$
\begin{align}
\mathcal{L}(\theta) &= \frac{1}{2} E_{X,Y} (Y-g^N(X;\theta))^2.
\end{align}
$$

We learn the parameters $\theta$ using stochastic gradient descent. The choice of the learning rate is theoretically linked to the $\gamma$ parameter. In particular, the theory developed in [8] suggests that for this case the learning rate should be chosen to be of the order of

$$
\alpha_{N,\gamma}=\frac{\alpha}{N^{2(1-\gamma)}},
$$

for $\alpha$ a constant of order one.

The goal of the paper [8] is to study  the performance of neural networks scaled by $1/N^{\gamma}$ with $\gamma\in [1/2, 1]$. 

The theoretical results of [8] derive an asymptotic expansion of the neural network's output with respect to $N$. This expansion demonstrates the effect of the choice of  $\gamma$ on bias and variance. In particular, for large and fixed $N$, the variance goes down monotonically as $\gamma$ increases to $1$.

The numerical results of [8], done on MNIST [10] and CIFAR10 [9] datasets, demonstrate that train and test accuracy monotonically increase as $\gamma$ increases to $1$. 

The conclusion is that being close to the mean-field normalization $\gamma=1$ is clearly the optimal choice!! But, for this to be realized the learning rate has to be chosen appropriately based on a theoretically informed choice (see for example the discussion above in the case of shallow neural networks).

Below are two numerical examples. **These numerical studies were done without further parameter tuning**.  The first example is on the MNIST data set [10] with the cross-entropy loss function with $N=3000$ hidden units (taken from [8]). 

![plot_mnist_ce_h3000_e500_b20_test](https://user-images.githubusercontent.com/106413949/172763587-1c41126e-368a-4f5f-8ab1-5c1b917dcc23.png)

The second example is an implementation of the same idea to a Convolutional Neural Network  (CNN) applied to the CIFAR10 dataset [9] (see [8] for details). The effect of the scaling is even more apparent here. 

![plot_cnn_cifar10_h3000_e1000](https://user-images.githubusercontent.com/106413949/172856057-dd0087bb-1d3a-4629-9b99-bd5ff1769185.png)

***These numerical studies were done without further parameter tuning. Further parameter tuning will improve the accuracy more. However, the point here is that*** $\gamma \rightarrow 1$ ***is a theoretically informed optimal choice (when paired with the correct learning rate choice) that allows the user to immediately achieve great accuracy without the need for further parameter tuning!!***







**References**

[1] L. Chizat, and F. Bach. On the global convergence of gradient descent for over-parameterized models
using optimal transport. Advances in Neural Information Processing Systems (NeurIPS). pp. 3040-3050,
2018.

[2] X. Glorot and Y. Bengio. Understanding the diffculty of training deep feedforward neural networks.
Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-
256. 2010.

[3] S. Mei, A. Montanari, and P. Nguyen. A mean field view of the landscape of two-layer neural networks
Proceedings of the National Academy of Sciences, 115 (33) E7665-E767, 2018.

[4] G. M. Rotskoff and E. Vanden-Eijnden. Neural Networks as Interacting Particle Systems: Asymptotic
Convexity of the Loss Landscape and Universal Scaling of the Approximation Error. arXiv:1805.00915,
2018.

[5] J. Sirignano and K. Spiliopoulos. Mean Field Analysis of Neural Networks: a law of large numbers.
SIAM Journal on Applied Mathematics, 80(2), 725-752, 2020.

[6] J. Sirignano and K. Spiliopoulos. Mean Field Analysis of Neural Networks: A Central Limit Theorem.
Stochastic Processes and their Applications, 130(3), 1820-1852, 2020.

[7] J. Sirignano and K. Spiliopoulos. Mean Field Analysis of Deep Neural Networks. Mathematics of
Operations Research, 47(1):120-152, 2021.

[8] K. Spiliopoulos and Jiahui Yu, Normalization effects on shallow neural networks and related asymptotic expansions, 2021, AIMS Journal on Foundations of Data Science, June 2021, Vol. 3, Issue 2, pp. 151-200.

[9] A. Krizhevsky Learning Multiple Layers of Features from Tiny Images, Technical Report, 2009.

[10] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition.
Proceedings of the IEEE, 86(11):2278-2324, 1998.
