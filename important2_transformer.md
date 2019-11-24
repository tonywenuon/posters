
# 【重要系列 1】生成模型 Transformer

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是 Transformer？
* Self-Attention 和 Multi-Head 机制是什么？
* 如何用 Keras 实现一个 Transformer 模型？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

**Motivation:** 在 Transformer 被提出之间，基本上 RNN 架构统治了 sequence-to-sequence 模型。尤其在 Attention 机制被提出后，Seq2Seq with Attention 得到了广泛的应用（详细原理可参见[https://zhuanlan.zhihu.com/p/87961308](https://zhuanlan.zhihu.com/p/87961308)）。虽然 LSTM 或者 GRU 有记忆能力，一定程度上能够获取长距离记忆，但是一旦距离过长，那么他的作用就有限了。举个例子来说，比如原句子为 “The animal didn't cross the road because it was too tired.”，在这个句子里 `it` 指代的就是 `animal`。如果用 LSTM 来计算，那么 `animal` 的语义信息，要通过 5 次计算（计算5个词 didn't cross the road because）后才能到达 `it`，那么这个时候 `animal` 的信息还剩多少呢？这个不好说，但是可以肯定的是中间经过了 5 次的遗忘门（forget gate）会有信息损耗。那么有没有一种方式，能够让 `it` 和 `animal` 直接进行交互，不通过中间那么多层呢？当然有，这就该 Transformer 出马了。

## 1. 什么是 Transformer？

Transformer，大家普遍翻译成变形金刚，我觉得这个翻译还挺有意思的，哈哈。言归正传，Transformer 实际上是一种架构，从架构上并体现不出来他为什么起名叫 Transformer。在 Transformer 中最重要的两个结构是 Self-Attention 和 Multi-Head。下面我们依据架构图逐一介绍。

![](https://github.com/tonywenuon/posters/blob/master/images/important2/transformer.png?raw=true)

如图左边部分是 Transformer 的 Encoder，右边是 Decoder。Encoder 主要有两层：Multi-Head Attention 和一个前向网络。其中的 `Add & Norm` 是 Residual Connection 和 Layer  Normalization。这里，前向网络，Residual Connection 和 Layer  Normalization 并不是本文的重点，我会在介绍完 Self-Attention 和 Multi-Head 之后再介绍一下他们的作用。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTIxNzkyMDY5NSwtMTA5NDMwMTA3NSw4OD
A3MjQxNTEsMTYzNDI2OTkxNiwxNTY5OTA5Mzc0LDE3Mjg2ODY2
NzQsMTc0MDYxNTk2MV19
-->