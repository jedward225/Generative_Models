# Notes for Deep Generative Model Learning

*(the whole notebook was written by JEd. And the DGM here merely focus on the instances on image generation)*

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207120439708.png" alt="image-20250207120439708" style="zoom: 67%;" />

![img](https://pica.zhimg.com/v2-3e63382aba8a29b05c2a8a6a7db3f82c_1440w.png)

## 0 - Pre-knowledge

- Transformer
- Resnet
- RNN & LSTM
- Maximum Likelihood Estimation (MLE)
- Divergence
- Basic Machine Learning (*e.g.* Gradients)
- Monte Carlo Estimation
- Stochastic Differential Equation (SDE) (? not necessary better to know)
- CLIP



## 1 - Auto-Regressive Model (ARM)

### Principle

Suppose there is a high-dimensional random variable $X$, where $x_i$ represents the little pixel points within it. According to the chain rule, we know that $p(x)$ can be calculated by this fomular:
$$
p(x) = p(x_1) \prod_{d=2}^D p(x_d | X_{<d})
$$
Under the autoregressive assumption, generation problem has become a **sequence problem**.

### Examples

#### Entire Linear Regression

The MINIST Task, we assume
$$
& p(x) = p_{\text{CPT}}(x_1; \alpha^{(1)})p_{\text{logit}}(x_2 | x_1; \alpha^{(2)})p_{\text{logit}}(x_3 | x_1,x_2; \alpha^{(3)}) \cdots p_{\text{logit}}(x_D | x_{<D}; \alpha^{(D)})
\\ \text{where},  &  p_{\text{CPT}}(X_1 = 0; \alpha^{(1)}) = \alpha^{(1)}, p_{\text{CPT}}(X_1 = 0; \alpha^{(1)}) = 1 - \alpha^{(1)} 
\\ & p_{\text{logit}}(X_2 = 1 | x_1; \alpha^{(2)}) = \sigma(\alpha_0^{(2)} +\alpha_1^{(2)}x_1)
$$

Without NN, in this kind of modeling of black and white images, the pixel computation entirely relies on linear regression.

<img src="https://pic2.zhimg.com/v2-afb00056167ee19c596c3d51c1368383_1440w.jpg" alt="img" style="zoom:50%;" />

For multi-channel images, autoregressive generation follows the order of RGB. The probability for the i-th pixel could become: 
$$
p(x_i) = p\left(x_{i, R} | x_{<i} \right) p\left(x_{i, G} | x_{<i}, x_{i, R}\right) p\left(x_{i, B} | x_{<i}, x_{i, R}, x_{i, G}\right)
$$
<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207124752266.png" alt="image-20250207124752266" style="zoom: 50%;" />

#### RNN & PixelRNN (×)

- Recap of RNN [Recurrent Neural Network Regularization](https://arxiv.org/pdf/1409.2329)
- PixelCNN Network Structure
- “Masked Convolution” & “BlindSpot”

> 由于像素本身的性质，对图片生成进行scaling或者rotation的操作是得额外学习的。



## 2 - Variational Auto-Encoders (VAE)

### Mixture of Gaussians

- A simple family of multi-modal distributions
- treat unimodal Gaussians as **basis (or component) distributions**
- superpose multiple Gaussians via **convex combination**

$$
p(x) = \sum_{k = 1}^{K} \pi_k \mathcal{N}(x|\mu_k, \sigma_k^2)
$$

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207131858240.png" alt="image-20250207131858240" style="zoom:50%;" />

![image-20250207132914624](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207132914624.png)

#### MLE for Mixture of Gaussians

$$
\theta = (\pi, \mu, \Sigma)
\\ \mathcal{L}(\mu, \Sigma) = \log p(\mathcal{D} | \pi, \mu, \Sigma) = \sum_{n = 1}^N \log\left(\sum_{k=1}^K \pi_{k} \mathcal{N}(x | \mu_k, \Sigma_k)\right)
$$

Since this is quite complicated, we can estimate it through a heuristic procedure (Expectation-Maximization Algorithm).
$$
& \frac{\partial \mathcal{L}}{\partial \mu_k} = 0 \\
\Rightarrow & \sum_{n=1}^N\frac{\pi_k\mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)} \Sigma_k^{-1}(x_n - \mu_k) = 0 \\
\Rightarrow & \mu_k = \frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})x_n, N_k = \sum_{n=1}^N\gamma(z_{nk})
\\
\\ & \frac{\partial \mathcal{L}}{\partial \Sigma_k} = 0 \\
\Rightarrow & \Sigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk})(x_n - \mu_k)(x_n - \mu_k)^T
\\
\\ & \frac{\partial \mathcal{L}}{\partial \pi_k} = 0 \\
\Rightarrow & \sum_{n=1}^N\frac{\mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)} + \lambda = 0 \\
\Rightarrow & \pi_k = \frac{N_k}{N}
$$
Since, the set of couple conditions is
$$
\mu_k = \frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})x_n
\\ N_k = \sum_{n=1}^N\gamma(z_{nk})
\\ \Sigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk})(x_n - \mu_k)(x_n - \mu_k)^T
\\ \pi_k = \frac{N_k}{N}
$$
And the key factor to get them coupled is
$$
\gamma(z_{nk}) = \frac{\pi_k\mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}
$$
Then, we can iterate the function.

> - **E-step**: estimate the responsibilities
>   $$
>   \gamma(z_{nk}) = \frac{\pi_k\mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}
>   $$
>
> - **M-step**: re-estimate the parameters
>   $$
>   \mu_k = \frac{1}{N_k}\sum_{n = 1}^N \gamma(z_{nk})x_n \\
>   \Sigma_k = \frac{1}{N_k} \sum_{n = 1}^N \gamma(z_{nk})(x_n - \mu_k)(x_n - \mu_k)^T \\
>   \pi_k = \frac{N_k}{N}
>   $$
>   So, **initialization plays a key role to succeed!**

### Latent variable model

Latent variable models take an indirect approach to describing a probability distribution $P(x)$ over a multi-dimensional variable $x$. Instead of directly writing the expression for $P(x)$, they model a joint distribution $P(x, z)$ of the data $x$ and an unobserved hidden or latent variable $z$. They then describe the probability of $P(x)$ as a marginalization of this joint probability so that:
$$
P(x) = \int P(x, z) \text{d}z = \int P(x|z)P(z) \text{d}z \text{, where }P(z) \text{ is the prior}.
$$
<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207120846077.png" alt="image-20250207120846077" style="zoom:67%;" />

#### The advantages of latent model

- Expressiveness power (e.g, single modal to muki modain MoG)
- Dimensionality reduction, structures and interpretability
- Conditional sampling and manipulation

Latent Representation is very powerful.(credit to [Lecture 7-8 From Autoencoder to VAE 110mins](https://deep-generative-models.github.io/files/ppt/2022/Lecture 7-8 From Autoencoder to VAE.pdf), [了解Sora背后的原理，来学习深度生成模型！(第2讲)](https://www.bilibili.com/video/BV17D421W7Ej/?spm_id_from=333.337.search-card.all.click&vd_source=f58d5b48f899e5f71e209e54735b9eac))

![Vanilla Autoencoder](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207132306659.png)

#### Evidence lower bound (ELBO)

- Jensen's inequality
  $$
  \log \mathbb{E}_{p(x)}[x] \ge \mathbb{E}_{p(x)}[\log x]
  $$

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207145635003.png" alt="image-20250207145635003" style="zoom: 50%;" />

- Now, let's proceed with some derivations.

$$
\log p(x) & = & \log \int p(x, z) \text{d}z \\
 & = & \log \int \frac{p(x, z)q_{\phi}(z|x)}{q_{\phi}(z|x)}  \text{d}z \\
 & = & \log \mathbb{E}_{q_{\phi}(z | x)} \left[\frac{p(x, z)}{q_{\phi}(z | x)}\right]\\
 & \geq &\mathbb{E}_{q_{\phi}(z | x)} \log\left[\frac{p(x, z)}{q_{\phi}(z | x)}\right] \\
 & = & \text{ELBO}[\theta, \phi]
$$
Here, $q_{\phi}(z|x)$ is a flexible approximate variational distribution with parameters $\phi$ that we seek to optimize. Additionally, in practice, $\theta$ is the parameter of the distribution $q(z)$. And the ELBO can also be written as:
$$
\text{ELBO}[\theta, \phi] = \int q(z | \theta) \log\left[\frac{P(x, z | \phi)}{q(z | \theta)}\right] \text{d}z
$$
<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207153939888.png" alt="image-20250207153939888" style="zoom: 50%;" />

Here are some notes about the ELBO:

- The ELBO is tractable because $p(x, z)$ and $q_\phi(z|x)$ are defined manually.

- The EBLO is valid for any $q$, but we optimize $p$ to obtain a tight lower bound.

- Maximizing ELBO is equivalent to minimize $KL(q_\phi(z|x)\ ||\ p(z|x))$, Variational Inference(VI).
  $$
  \log p(x) & = & \log \int p(x, z) \text{d}z \\
   & = & \mathbb{E}_{q_{\phi}(z | x)} [\log p(x)] \\
   & = & \mathbb{E}_{q_{\phi}(z | x)} \left[\log \frac{p(x, z)}{p(z | x)}\right] \\
   & = & \mathbb{E}_{q_{\phi}(z | x)} \left[\log \frac{p(x, z)}{q_\phi(z | x)}\right] + \mathbb{E}_{q_{\phi}(z | x)} \left[\log \frac{q_\phi(z | x)}{p(z | x)}\right] \\
    & = & \mathbb{E}_{q_{\phi}(z | x)} \left[\log \frac{p(x, z)}{q_\phi(z | x)}\right] + D_{\text{KL}}(q_\phi(z|x)\ ||\ p(z|x)) \\
    & \geq &\mathbb{E}_{q_{\phi}(z | x)} \log\left[\frac{p(x, z)}{q_{\phi}(z | x)}\right] (\text{tips: } D_{\text{KL}}\geq0)
  $$

- The ELBO is *tight* when, for a fixed value of $\phi$, the ELBO and the likelihood function coincide.

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207150041745.png" alt="image-20250207150041745" style="zoom:67%;" />

#### Variational Inference (×)

Variational inference methods in Bayesian inference and machine learning are techniques which are involved in approximating intractable integrals. 

https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture13.pdf

https://ashkush.medium.com/variational-inference-gaussian-mixture-model-52595074247b

<img src="https://odie2630463.github.io/media/C8CD4EAE-330B-4369-BB79-BA974B69EA45.png" alt="C8CD4EAE-330B-4369-BB79-BA974B69EA45" style="zoom: 50%;" />



#### Variational approximation

The ELBO is tight when $q(z|\theta)$ is the posterior $P(z|x, \phi)$.

In principle, we can compute the posterior using Bayes’ rule:
$$
P(z|x, \phi) = \frac{P(x|z, \phi)P(z)}{P(x|\phi)}
$$
Here, 

- $P(z|x, \phi)$ is the **posterior probability**, representing the distribution of the latent variable $z$ given the observed data $x$ and parameters $\phi$. 
- $P(x|z, \phi)$ is the **likelihood**, indicating the probability of observing data $x$ given the latent variable $z$ and parameters $\phi$. 
- $P(z)$ is the **prior probability**, expressing the initial belief or distribution about the latent variable $z$ before seeing the data $x$. 
- $P(x|\phi)$ is the **marginal likelihood** or **evidence**, which can be obtained by integrating the joint probability $P(x,z|\phi)$ over all possible values of $z$, i.e., $P(x|\phi) = \int P(x|z, \phi)P(z) dz$. This step effectively marginalizes out $z$, making it independent of $z$ but dependent on $x$ and $\phi$.

But in practice, this is intractable because we can’t evaluate the data likelihood in the denominator.

One solution is to make a variational approximation: we choose a simple parametric form for $q(z|θ)$ and use this to approximate the true posterior. We can choose a multivariate normal distribution with mean $\mu$ and diagonal covariance $\Sigma$. This will not always match the posterior well but often performs well. During training, we will find the normal distribution that is “closest” to the true posterior $P(z|x)$.

Since the optimal choice depends on the data example $x$, the variational approximation should do the same, so we choose:
$$
q(z | x, \theta) = \text{Norm}_z\left[g_\mu[x, \theta], g_\Sigma[x, \theta]\right]
$$
where $g[x,\theta]$ is a second neural network with parameters $θ$ that predicts the mean $\mu$ and variance $\Sigma$ of the normal variational approximation.

#### Structure of VAE

According to *Monte Carlo estimate*, For any function $a[·]$ we have:
$$
\mathbb{E}_z[a[z]] = \int a[z]q(z|x, \theta)\text{d}z \approx \frac{1}{N}\sum_{n=1}^N a[z_n^*]
$$
where $z_n^*$ is the n-th sample from $q(z|x, \theta)$.

And thus, we can better know how to calculate ELBO from the perspective of algorithm:
$$
\text{ELBO} & \approx & \frac{1}{N}\sum_{n=1}^N\log\left[P(x|z_n^{*},\phi)\right] - D_{\text{KL}}\left[q(z|x, \theta)\ ||\ P(z)\right] \\
& = & \frac{1}{N}\log\left[\prod_{n = 1}^NP(x|z_n^{*},\phi)\right] - D_{\text{KL}}\left[q(z|x, \theta)\ ||\ P(z)\right]
$$
Notice that the second term is the KL-divergence between the variational distribution $q(z|x, \theta) = \text{Norm}_z[\mu, \Sigma]$ and the prior $P(z) = \text{Norm}_z[0, I]$. **The KL-divergence between two normal distributions can be calculated in closed form**. It is given by:
$$
D_{\text{KL}}\left[q(z|x, \theta)\ ||\ P(z)\right] = \frac{1}{2}\left(\tr[\Sigma] + \mu^T\mu - D_z - \log\left[\det[\Sigma]\right]\right)
$$
where $D_z$ is the dimensionality of the latent space. So we have the loss function:
$$
\text{ELBO} \approx \frac{1}{N}\sum_{n=1}^N\log\left[P(x|z_n^{*},\phi)\right] - \frac{1}{2}\left(\tr[\Sigma] + \mu^T\mu - D_z - \log\left[\det[\Sigma]\right]\right) & (*)\\
$$
and our objective is:
$$
\arg\max\limits_{\phi, \theta} \frac{1}{N}\sum_{n=1}^N\log\left[P(x|z_n^{*},\phi)\right] - D_{\text{KL}}\left[q(z|x, \theta)\ ||\ P(z)\right] & (**)\\
$$
![image-20250207160152249](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207160152249.png)

It should now be clear why this is called a *variational autoencoder*. It is variational because it computes a Gaussian approximation to the posterior distribution. It is an autoencoder because it starts with a data point $x$, computes a lower-dimensional latent vector $z$ from this, and then uses it to recreate the data point $x$ as closely as possible. In this context, the mapping from the data to the latent variable by the network $g[x, \theta]$ is called the *encoder*, and the mapping from the latent variable to the data by the network $f[x, \theta]$ is called the *decoder*.

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207164245725.png" alt="image-20250207164245725" style="zoom: 50%;" />

#### Policy Gradient (×)



#### The Reparameterization Trick

The reparameterization trick rewrites a random variable as a deterministic function of a noise variable; this allows for the optimization of the non-stochastic terms through gradient descent. For example, samples from a normal distribution $x\sim \mathcal{N}(x; \mu, \sigma^2)$ with arbitrary mean $\mu$ and variance $\sigma^2$ can be rewritten as:
$$
x=\mu+\sigma\epsilon & \text{  with  } \epsilon \sim \mathcal{N} (\epsilon  ; 0,I) \\
$$
By the reparameterization trick, sampling from an arbitrary Gaussian distribution can be performed by sampling from a standard Gaussian, scaling the result by the target standard deviation, and shifting it by the target mean.

![image-20250207162353303](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207162353303.png)

#### AEs & VAE variants (×)

> 在深度学习的早期，自动编码器（Autoencoder, AE）被广泛用于无监督学习中，尤其是用于降维和特征学习。自动编码器的工作机制是通过一个编码器将输入数据压缩为一个潜在表示，再通过解码器从该潜在表示重建原始数据。虽然自动编码器在重构数据时有一定的能力，但它并不擅长生成新的数据点。这是因为传统的自动编码器将输入数据映射到一个**固定的潜在空间**，没有明确地对潜在空间的分布进行建模。
>
> 所以从结果上来讲，AE的泛化能。

- Vanilla Autoencoder
- Denoising Autoencoder
- Sparse Autoencoder
- Contractive Autoencoder
- Stacked Autoencoder
- Variational Autoencoder (VAE)

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207163919994.png" alt="image-20250207163919994" style="zoom:50%;" />

### VAE VS. ARM

![image-20250207164339215](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207164339215.png)

![image-20250207164618654](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207164618654.png)

### Markovian Hierarchical VAE

A Hierarchical Variational Autoencoder (HVAE) is a generalization of a VAE that extends to multiple hierarchies over latent variables.

![image-20250209181523618](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250209181523618.png)

Intuitively, and visually, this can be seen as simply stacking VAEs on top of each other, as depicted in Figure 2; another appropriate term describing this model is a Recursive VAE. Mathematically, we represent the joint distribution and the posterior of a Markovian HVAE as:
$$
p(x, z_{1:T}) = p(z_T)p_\theta(x|z_1)\prod_{t=2}^Tp_\theta(z_{t-1}|z_t) \\
q_\phi(z_{1:T}|x) = q_\phi(z_1 | x) \prod_{t=2}^T q_\phi(z_t|z_{t-1})
$$
And the ELBO can be extended to be:
$$
\log p(x) & = & \log\int p(x, z_{1:T})\text{d}z_{1:T} \\
& = & \log\int \frac{p(x, z_{1:T}) q_\phi(z_{1:T})}{q_\phi(z_{1:T})}\text{d}z_{1:T} \\
& = & \log \mathbb E_{q_\phi(z_{1:T} | x)}\left[\frac{p(x, z_{1:T})}{q_\phi(z_{1:T} | x)}\right] \\
& \geq & \mathbb E_{q_\phi(z_{1:T} | x)} \left[\log\frac{p(x, z_{1:T})}{q_\phi(z_{1:T} | x)}\right] \\
& = & \mathbb E_{q_\phi(z_{1:T} | x)} \left[\log\frac{p(z_T)p_\theta(x|z_1)\prod_{t=2}^Tp_\theta(z_{t-1}|z_t)}{ q_\phi(z_1 | x) \prod_{t=2}^T q_\phi(z_t|z_{t-1})}\right]
$$

## 3 - Normalizing Flow Model (×)





## 4 - Generative Adversarial Networks (GAN)

### High likelihood = good sample quality?

Case 1: Optimal generative model will give best sample quality andhighest test log-likelihood

For imperfect models, achieving high log-likelihoods might not alwaysimply good sample quality, and vice-versa (Theis et al., 2016)

![image-20250207165230438](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207165230438.png)

![image-20250207165244024](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207165244024.png)

### Structure of GAN

#### Loss Function

GAN’s architecture consists of two neural networks:

1. **Generator(G)**: creates synthetic data from random noise to produce data so realistic that the discriminator cannot distinguish it from real data.

   The Generator is a deep neural network (DNN) that takes random noise as input to generate realistic data samples, learning the underlying data distribution by adjusting its parameters through backpropagation (BP). Its loss function is:
   $$
   \mathcal{L}_G=\mathbb{E}_{x\sim p_{z}}[\log(1 - D(G(z)))] \\
   G^* = \min_\limits{G} \mathbb{E}_{x\sim p_{z}}[\log(1 - D(G(z)))]
   $$
   Where,

   - $\mathcal{L}_G$ measure how well the generator is fooling the discriminator.
   - $z$ is random noise sampled from the noise distribution $p_z$.

2. **Discriminator(D)**: acts as a critic, evaluating whether the data it receives is real or fake.

   The Discriminator acts as a binary classifier, distinguishing between real and generated data. This loss incentivizes the discriminator to accurately categorize generated samples as fake and real samples with the following equation:
   $$
   \mathcal{L}_D=-\mathbb{E}_{x\sim p_{data}}[\log D(x)]-\mathbb{E}_{x\sim p_{z}}[\log(1 - D(G(x)))] \\
   D^* = \max_\limits{D} \mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{x\sim p_{z}}[\log(1 - D(G(x)))]
   $$
   Where,

   - $\mathcal{L}_D$ assesses the discriminator’s ability to discern between produced and actual samples.
   - $x$ is the real data sample from the distribution $p_{data}$.

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207174417583.png" alt="image-20250207174417583" style="zoom: 80%;" />

The training process of a GAN (Generative Adversarial Network) is a minimax game, where the goals of the generator and the discriminator are opposed:
$$
\min_\limits{G} \max_\limits{D} V(D, G) = \min_\limits{G}\max_\limits{D} \mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{x\sim p_{z}}[\log(1 - D(G(x)))]
$$
Optimal disciminator:
$$
D_G^* = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}
$$
For the optimal disciminator $D_G^*(·)$, we have
$$
V(G, D_G^*(x)) & = & \mathbb{E}_{x\sim p_{data}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}\right]+\mathbb{E}_{x\sim p_{z}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}\right] \\
& = & \mathbb{E}_{x\sim p_{data}}\left[\log \frac{p_{data}(x)}{\frac{p_{data}(x) + p_G(x)}{2}}\right]+\mathbb{E}_{x\sim p_{z}}\left[\log \frac{p_{data}(x)}{\frac{p_{data}(x) + p_G(x)}{2}}\right] - \log 4 \\
& = & D_{\text{KL}}\left[p_{data}, \frac{p_{data}(x) + p_G(x)}{2}\right] +D_{\text{KL}}\left[p_{G}, \frac{p_{data}(x) + p_G(x)}{2}\right] - \log 4 \\
& = & 2D_{\text{JSD}}\left[p_{data}, p_{G}\right] - \log 4
$$
<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207174319391.png" alt="image-20250207174319391" style="zoom: 67%;" />

#### Jesen-Shannon divergence

![image-20250207174050893](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207174050893.png)

#### Evaluation

GANs are assessed based on the ***\*quality, diversity, and realism\**** of their generated samples. Common evaluation metrics include:

**Fréchet Inception Distance (FID)** – Measures similarity to real images.

***\*Inception Score (IS)\**** – Evaluates the variety and clarity of generated images.

Evaluation of Generative Models: Sample Quality • Known Ground Truth • SRGAN •-MSE PSNR SSIM • Unknown Ground Truth • DRIT • StarGAN •-Classification •-IS FID KID •-LPIPS • Human Evaluation • Ranking v.s. Contrast • Tools: AMT

[Lecture 16-17 Evaluation - Sampling Quality 50-80min](https://deep-generative-models.github.io/files/ppt/2022/Lecture 16-17 Evaluation - Sampling Quality.pdf)

### Optimization challenges (略)

- **Training instability** – Balancing the generator and discriminator is tricky.
- **Mode collapse** – The generator may produce only a few types of outputs instead of diverse samples.
- **High computational cost** – Requires powerful hardware for training.

### Further Variants (×)



## 5 - Energy-Based Model (EBM) (×)





## 6 - Diffusion Probabilistic Model (DPM)

![img](https://pic1.zhimg.com/v2-3ce40580db330cd3d35fb4db24aa2438_r.jpg)

### An Overview From Markov Chain

![image-20250207180505465](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207180505465.png)

Diffusion process gradually injects noise to data

Described by a Markov chain:
$$
q(x_0, \cdots, x_N) = q(x_0)q(x_1 | x_0) \cdots q(x_N | x_{N-1})
$$
![image-20250207180851795](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250207180851795.png)

Diffusion process in the reverse direction ⇔ denoising proces

Reverse factorization (We need an approximate diffusion process in the reverse direction):
$$
q(x_0, \cdots, x_N) = q(x_0 | x_1)\cdots q(x_{N-1}|x_N)q(x_N)
$$

> 1. 从右向左是encoder过程，是一个无参数的 $q(x_t|x_{t−1})$，即这是一个纯粹人为的过程（比如从原始的清晰图片每次按照高斯分布进行一个映射）。
> 2. 从左往右是decoder过程，是一个带参数的 $p_\theta(x_{t−1}|x_t)$ 。其并非像VAE那样直接预测 $\hat x$ ，而是预测高斯噪声，在decode过程中逐渐减去高斯噪声，还原出清晰的图像。
> 3. 在diffusion model 中隐变量 $z$ （在图中是 $x_N(\text{or usually } x_T)$ ）和原始图片的维度是一样大的（但是理论上还是简单的（因为很像高斯噪声））。

Let's take a recap of two formulation of MHVAE:

> $$
> p(x, z_{1:T}) = p(z_T)p_\theta(x|z_1)\prod_{t=2}^Tp_\theta(z_{t-1}|z_t) \\
> q_\phi(z_{1:T}|x) = q_\phi(z_1 | x) \prod_{t=2}^T q_\phi(z_t|z_{t-1})
> $$

The VDM posterior is the same as the MHVAE posterior, but can now be rewritten as:
$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})
$$
with the assumption of:
$$
q(x_t | x_{t-1}) = \mathcal N(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)
$$
this assumption can be rewritten into $x_t = \alpha_t x_{t-1}+\beta_t\varepsilon_t, \varepsilon_t \in \mathcal N(0, I), \alpha_t, \beta_t > 0 \wedge \alpha_t^2 + \beta_t^2 = 1$, with $\beta_t$ close to o. In the case,
$$
x_t & = & \alpha_t x_{t-1} + \beta_t \varepsilon_t \\
& = & \alpha_t( \alpha_{t-1} x_{t-2} + \beta_{t-1} \varepsilon_{t-1}) + \beta_t \varepsilon_t \\
& = & \cdots \\
& = & (\alpha_t\cdots\alpha_1)x_0 + (\alpha_t\cdots\alpha_2)\beta_1\varepsilon_{1} + (\alpha_t\cdots\alpha_3)\beta_2\varepsilon_{2} + \cdots + \alpha_t \beta_{t-1}\varepsilon_{t-1} + \beta_t\varepsilon_{t} \\
\text{key!}& = & (\alpha_t\cdots\alpha_1)x_0 + \sqrt{1 - (\alpha_t\cdots\alpha_1)^2} \bar \varepsilon_t, \bar \varepsilon_t \sim \mathcal N(0, I)
$$
And,
$$
p(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t) \\
\text{where, } p(x_T) = \mathcal N(x_T; 0, I)
$$
This means that, our encoder distributions $q(x_t|x_{t-1})$ are no longer parameterized by $\phi$, as they are completely modeled as Gaussians with defined mean and variance parameters at each timestep. Therefore, in a VDM, we are only interested in learning conditionals $p_\theta(x_{t-1} | x_t)$. After optimizing the VDM, the sampling procedure is as simple as sampling Gaussian noise from $p(x_T)$ and iteratively running the denoising transitions for T steps to generate a novel $x_0$.

ELBO can be derived as:
$$
\log p(x) & = & \log \int p(x_{0:T})\text{d}x_{1:T} \\
& = & \log \int \frac{p(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} \text{d}x_{1:T} \\
& = & \log \mathbb E_{q(x_{1:T}|x_0)}\left[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] \\
& \ge & \mathbb E_{q(x_{1:T}|x_0)}\log\left[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] \\
& = & \mathbb E_{q(x_{1:T}|x_0)}\log\left[\frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)}{\prod_{t=1}^T q(x_{t} | x_{t - 1})}\right] \\
& = & \mathbb E_{q(x_{1:T}|x_0)}\log\left[\frac{p(x_T) p_\theta(x_{0} | x_1) \prod_{t=1}^{T-1} p_\theta(x_{t} | x_{t+1})}{q(x_{T} | x_{T - 1})\prod_{t=1}^{T-1} q(x_{t} | x_{t - 1})}\right] \\
& = & \mathbb E_{q(x_{1:T}|x_0)}\log\left[\frac{p(x_T) p_\theta(x_{0} | x_1)}{q(x_{T} | x_{T - 1})}\right] + \mathbb E_{q(x_{1:T}|x_0)}\log\left[\frac{\prod_{t=1}^{T-1} p_\theta(x_{t} | x_{t+1})}{\prod_{t=1}^{T-1} q(x_{t} | x_{t - 1})}\right] \\
& = & \mathbb E_{q(x_{1:T}|x_0)}\log\left[ p_\theta(x_{0} | x_1)\right] + \mathbb E_{q(x_{1:T}|x_0)}\log\left[\frac{p(x_T)}{q(x_{T} | x_{T - 1})}\right] + \mathbb E_{q(x_{1:T}|x_0)}\sum_{t=1}^{T-1}\log\left[\frac{p_\theta(x_{t} | x_{t+1})}{q(x_{t} | x_{t - 1})}\right]
$$


![image-20250209184342763](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250209184342763.png)



### Encoder (Forward Process of DDPM)



### Decoder (Backward Process of DDPM)



### DDIM



### Architecture of Stable Diffusion (credit to *Umar Jamil*, Coding Stable Diffusion  form scratch in PyTorch)

![image-20250210084637269](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250210084637269.png)

![image-20250210084838538](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250210084838538.png)







## 7 - Score-based Model

![image-20250222232119338](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250222232119338.png)

![image-20250222232213309](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250222232213309.png)

![image-20250222232243025](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250222232243025.png)

## ? - Reference

**Understanding Deep Learning** by Simon J.D. Prince

[Generative Adversarial Network (GAN) - GeeksforGeeks](https://www.geeksforgeeks.org/generative-adversarial-network-gan/)

https://yang-song.net/blog/2021/score/

[Diffusion Probabilistic Models: Theory and Applications - Fan Bao](https://ml.cs.tsinghua.edu.cn/~fanbao/Application-DPM.pdf)

[扩散模型(Diffusion Model)首篇综述-Diffusion Models: A Comprehensive Survey of Methods and Applications - 知乎](https://zhuanlan.zhihu.com/p/562389931)

苏剑林. (Jun. 13, 2022). 《生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼 》[Blog post]. Retrieved from https://kexue.fm/archives/9119

