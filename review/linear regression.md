# linear regression


linear regression实质上是将已知的数据集通过相关的建模，训练出其数据相关预测的线性函数。

* 预测函数，$h(\theta) = \theta * X$
* costfunction J: $ J = 1/2m * \sum_{i=1}^{n}{(h(\theta) -y)^2}$ 计算模型与实际的误差。
* Gradient dencet：$J(\theta) := J(\theta) - \alpha*1/m\sum_{i=1}^{m}{(h(\theta) - y)*x^i} $  将各个feature:$\theta$进行训练，通过迭代，得到最先小的cost J，来实现最后的数据预测模型。

线性的函数与实际的数据并不能很好的实现对数据的反映，所以可以使用高阶的幂函数来建立模型。
