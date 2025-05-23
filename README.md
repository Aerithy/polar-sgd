# polar-sgd

SGD formula on node $i$: 

$$
\theta_{t + 1}^i = \theta_t^i - \eta \cdot g_t^i
$$

then: 

$$
\theta_{t + T}^i = \theta_t^i - \eta \cdot \sum_{n = 0}^{T - 1} g_{t + n}^i
$$

$\theta_t^i$ is the model parameters on node $i$ at batch $t$, $g_t^i$ is the gradient of the model parameters on node $i$ at batch $t$. $\eta$ is the learning rate.

For Local SGD,

$$
\begin{align}
    \theta_{t + T} & = \frac{1}{K} \sum_{i=0}^{K-1} \theta_{t + T}^i \\
    \theta_{t + T} & = \frac{1}{K} \sum_{i=0}^{K-1} \left(\theta_t^i - \eta \cdot \sum_{n = 0}^{T - 1} g_{t + n}^i \right) \\
    \theta_{t + T} & = \frac{1}{K} \sum_{i=0}^{K-1} \theta_t^i - \frac{1}{K} \sum_{i=0}^{K-1} \eta \cdot \sum_{n = 0}^{T - 1} g_{t + n}^i \\
    \theta_{t + T} & = \theta_t - \frac{\eta}{K}\sum_{i=0}^{K-1}\sum_{n = 0}^{T - 1} g_{t + n}^i \\
\end{align}
$$

As the above formula (1) shows, the global model parameters are calculated by averaging the local model parameters of time step $t + T$, which leads to strictly synchrounous updates. Strictly synchronous updates are not always desirable, cause a strictly synchronous update means communication cannot be overlapped with computation. When communication overhead is high, a not overlapped communication often leads to worse performance. 

We proposed a new method to update the global model parameters, which is called Polar SGD. 

In Polar SGD, we update the global model parameters by transmit gradient. The Architecture of Polar SGD is shown in the following figure. 

<img src="./overview.png">


We proposed a gradient prediction method to increase the communication and computation overlap. lets assume the prediction function is $P(g, \tau)$, which $g$ is the gradient and $\tau$ is the gradient of next $\tau$ time steps.

$$
\begin{align}
    \theta_{t + T}^i & = \theta_t^i - \eta \cdot P(g, T) \\
\end{align}
$$

So the global model update formula is: 

$$
\begin{align}
    \theta_{t + T} & = \frac{1}{K} \sum_{i=0}^{K-1} \theta_{t + T}^i \\
    \theta_{t + T} & = \frac{1}{K} \sum_{i=0}^{K-1} \left(\theta_t^i - \eta \cdot \sum_{n = 0}^{T - 1} g_{t + n}^i \right) \\
    \theta_{t + T} & = \frac{1}{K} \sum_{i=0}^{K-1} \theta_t^i - \frac{1}{K} \sum_{i=0}^{K-1} \eta \cdot \sum_{n = 0}^{T - 1} g_{t + n}^i \\
    \theta_{t + T} & = \theta_t - \frac{\eta}{K}\sum_{i=0}^{K-1}\sum_{n = 0}^{T - 1} g_{t + n}^i \\
\end{align}
$$