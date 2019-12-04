# 【BERT 系列 2】之 XLNet

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 自回归（AutoRegressive, AR）和自编码（AutoEncoder, AE）思想的区别？
* XLNet 的创作动机是什么？
* 什么是双流注意力机制（Two-steam Attention）？
* XLNet 和 BERT 的对比。

#### 重要文章
* <span id = "paper1">Paper 1</span>:[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)
* <span id = "paper2">Paper 2</span>:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

之所以把 XLNet 归到 BERT 系列是因为 XLNet 的逻辑是提升 BERT 模型天然的短板，弥补了 BERT 中的两个缺陷。

---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTMyOTg4NTA5NiwtNTUzODgwODM1LC0xNz
A4ODQ1Nzg2XX0=
-->