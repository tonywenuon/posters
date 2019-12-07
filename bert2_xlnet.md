# 【BERT 系列 2】之 XLNet

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 自回归（AutoRegressive, AR）和自编码（AutoEncoder, AE）思想的区别？
* XLNet 的研究出发点是什么？
* 什么是双流注意力机制（Two-steam Attention）？
* XLNet 和 BERT 的对比。

#### 重要文章
* <span id = "paper1">Paper 1</span>:[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)
* <span id = "paper2">Paper 2</span>:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

之所以把 XLNet 归到 BERT 系列是因为 XLNet 的逻辑是提升 BERT 模型天然的短板，弥补了 BERT 中的两个缺陷。加上在他们之后发布的文章很多也会拿他们俩来比较，我在这里也把他们分到同一个系列中。原文章里涉及到比较多的公式，我这里能省就省了。

### 1. 自回归（AutoRegressive, AR）和自编码（AutoEncoder, AE）
**自回归**，如果了解语言模型的话，自回归就很好理解了。拿预测一段文字序列来举例子，AR 的意思是依赖于前面的文字来生成后面的文字。那下面的例子来说，假设给一段文字 ：

> 完整文字段：知乎问答不回答专栏的文章很好很棒很赞
> 部分文字段：知乎问答不回答专栏的文章
> 需预测文字段：很好很棒很赞

现在的要求是给定 `部分分字段` 来预测 `需预测文字段`，那么就要根据 `知乎问答不回答专栏的文章` 先来预测 `很`，再根据 `知乎问答不回答专栏的文章很` 来预测 `好`。接下来依次类推，每次把预测出来的文字加入到 `部分文字段` 中，来预测下一个字。这是基本语言模型的原理。

**自编码**，自编码则和自回归不一样，他是在训练集中引入噪音，再来预测自己本身。他的思路是，在有噪音的训练集中把自己成功的还原，则这个模型就有去噪的能力，那么他表征自己的能力也就更强了。还是拿上面的例子来解释。

> 完整文字段：知乎问答不回答专栏的文章很好很棒很赞
> 部分文字段：知乎问答 [MASK] 回答专栏的 [MASK] 很好 [MASK] 很赞
> 需预测文字段：不  文章  很棒

可以看到，他是把原文字段中的某些字或词 mask 掉（通过占位符 [MASK]来替代原有文字），这就相当于去掉了部分有用信息而引入了部分噪音。其预测的目标是把这些去掉的信息还原回来。

### 2. XLNet 的出发点
正如上面介绍的，语言模型（Language Model，LM）属于自回归方式来训练，BERT 则正是属于自编码的训练方式。我们首先来说 BERT，基于上面的分析，BERT 存在两个问题。
**BERT 问题1**，在预测 [MASK] 位置的词的时候，每个 [MASK] 都是独立地预测，比如：

> 原句子：深圳是个大都市
> Mask 句子：[MASK] [MASK] 是个大都市

可以看到，在预测的时候，这两个 [MASK] 相当于是这样预测的，在给定 `是个大都市` 的条件下，预测 `深` 和 `圳` 的概率分别是：p(深 | 是个大都市) 和 p(圳 | 是个大都市)。而这显然是有缺陷的，没有考虑到 `深圳` 这两个词的关系（）

---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMjk3MDA1NTIsLTg0NDA3MzUyLDMwNj
cwMjg3OSwtMTM4MzkyMTM5MSwtNTUzODgwODM1LC0xNzA4ODQ1
Nzg2XX0=
-->