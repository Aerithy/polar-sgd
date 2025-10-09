# 关于本方法的设计和实现

本方法是本人的毕业设计中的其中一点，其目的是实现一个基于torch的创新并行策略，其核心思想是将模型参数按层分区，每次DP更新仅传输一部分分区梯度从而实现低通信并行。

细节如下：

1. 实现一个DP并行，该并行中包括了本方法的全部设计
2. DP并行内部采用PP并行的方式初步实现大型模型的并行训练
3. 将PP并行的模型分区和DP并行中涉及的模型分区组合起来

## 文件组织方式
首先看一下文件组织：

```
./parallelism  
├── __init__.py  
└── polar  
    ├── Technical_Explanation.md  
    ├── hooks.py  
    ├── util.py  
    └── wrapper.py  
```

hooks.py包括了许多本并行方法采用的 torch hook 函数，是本方法实现的核心。:rocket:  
wrapper.py主要是 torch DistributedDataParallelism (DDP) 的类似实现，用于包装模型并训练。 :sparkles:  
util.py主要包括了模型参数的分区方法等可以复用的通用方法。



## 模型分区方法
为了能够给模型很好的分区，传输对应分区的梯度并控制传输梯度时机，并且能够较好的兼容DP进程下的PP进程，我们这里使用torch.distributed.pipelining提供的方法


## Polar-SGD中的pipeline_model
我们的核心策略是在PolarWrapper中维护两个模型实例
1. self.model 原始的、完整的 nn.Module 模型，用于两个关键任务
    * 注册外部 Hook：PolarCommHook 会注册在从 self.model 分割出的 model_partitions 上，这保证了关于 Hook 的调度完全不受影响
    * 创建优化器：optimizer 会继续使用 self.model.parameters() 进行初始化，从而管理模型的所有参数
2. self.pipeline_model：一个由 torch.distributed.pipelining.pipeline API 创建的 Pipeline 对象，该对象封装了当前 DP 进程需要执行的流水线阶段。将用于：
    * 执行训练：在 train 方法中，我们将用 self.pipeline_model 替换 self.model 来执行前向和反向传播

*为什么可以这样做？*

torch.distributed.pipelining.pipeline 在创建流水线时，并不会复制模型参数，而是通过 tracing (FX Graph) 的方式引用原始模型的参数。这意味着，当 self.pipeline_model 在训练中计算出梯度时，这些梯度会直接累积在 self.model 对应参数的 .grad 属性上。因此，当 PolarCommHook 被触发时，它能像以前一样从 self.model 的参数中正确地收集到梯度。

这样，我们就实现了计算（PP）和外部通信（DP Hook）的解耦。:rocket:
