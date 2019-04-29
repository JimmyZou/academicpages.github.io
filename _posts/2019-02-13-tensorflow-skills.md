---
title: 'Python (Tensorflow) Implementation Skills Summary'
date: 2019-02-13
permalink: /posts/2019/02/2019-02-13-tensorflow-skills/
tags:
  - tensorflow
---
This blog is a summary of implementation skills in tensorflow for some special functions. The code provided is mostly test by myself. All these points take me much time to explore how to implement. So it is worth to summarize here.

---
**Self-define gradient for a layer using _tf.custom_gradient_**  
When the output (_y_) decribes a multi-variate orthogonal normal distribution with two means and standard deviations. In reinforcement learning, the continuous action can be sampled according to 
$$\pi(a|s) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\big(-\frac{(a-\mu)^2}{2\sigma^2}\big).$$ 
The gradient w.r.t $$\mu$$ and $$\sigma$$ in both dimensions is
$$\nabla_{\mu} \ln \pi(a|s) = \nabla_{\mu} [-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(a-\mu)^2}{2\sigma^2}] = \frac{a - \mu}{\sigma^2}$$ 
and 
$$\nabla_{\sigma} \ln \pi(a|s) = \nabla_{\sigma} [-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(a-\mu)^2}{2\sigma^2}] = \frac{ (a-\mu)^2 - \sigma^2}{\sigma^3}$$,
which is the initial input of the back-propagation in a neural network. The auto-gradient in TF cannot compute gradient like these. So we need to self-define a layer _sample_normal_dist_grad_layer_ with forward and backward propagation. Note that **grads = _y * dy** means that the initial input of this layer in back-propagation is **out** (the output of this final layer).

```
    import tensorflow as tf
    import numpy as np
    
    @tf.custom_gradient
    def sample_normal_dist_grad_layer(_y):
        mu = y[:, 0: 2]
        sigma = y[:, 2: 4]
        dist = tf.distributions.Normal(loc=0., scale=0.)
        sample = dist.sample(tf.shape(x)[0])
        out = sample * sigma + mu
        mu_loss = (out - mu) / (sigma ** 2)
        sigma_loss = ((out - mu) ** 2 - sigma ** 2) / (sigma ** 3)
        out = tf.concat([mu_loss, sigma_loss], axis=-1)
    
        def grad(dy):
            grads = _y * dy
            return grads
        return out, grad
    
    x = tf.placeholder(shape=[None, 2], name='input', dtype=tf.float32)
    y = tf.contrib.layers.fully_connected(x, 4, activation_fn=None)
    policy_loss = sample_normal_dist_grad_layer(y)
    grads_normal_dist = tf.gradients(policy_loss, y)
    
    opt_for_policy = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    vars_of_policy = tf.trainable_variables()
    grads_for_policy_and_vars = opt_for_policy.compute_gradients(policy_loss, vars_of_policy)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        b = sess.run([policy_loss, vars_of_policy, grads_for_policy_and_vars, grads_normal_dist, y], 
                     feed_dict={x: np.array([[1, 1]])})
        print('y', b[0])
        print('gradient_wrt_y', b[4])
        print('vars_of_policy', b[1][0], '\n', b[1][1])
        print('grads_for_policy_and_vars', b[2][0], '\n', b[2][1])
        print('grads_normal_dist', b[3])
```

---
**Sampling in categorical distribution using Gumbel softmax**  
Backpropagation is not possible through stochastic nodes (layers). We use the Gumbel-Max trick, which provides an efficient way to draw samples $$z$$ from the Categorical distribution with class probabilities $$\pi_i$$:
$$z=\text{one-hot} (\text{argmax}_{i}[g_i+\log\pi_i])$$.
The trick for correct backpropagation is this line: _y = tf.stop_gradient(y_hard - y) + y_.

```
def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    # trick for correct backpropagation
    y = tf.stop_gradient(y_hard - y) + y
  return y
```

---
**Ornstein-Uhlenbeck Action Noise**  
Simulating the Ornstein–Uhlenbeck process is based on [http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab](http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab). 
```
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
```

