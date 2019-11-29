
# 【BERT 系列 1】之 BERT 本尊

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是 BERT？
* BERT 最重要的贡献是什么？bidirectional 和 pre-train
* 如何用 BERT 来实现几个下游 NLP 任务？
* BERT 的局限性在哪里？
* BERT 用到了哪些数据集？以及数据集的简介和地址是什么？

#### 重要文章
* <span id = "paper1">Paper 1</span>:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

鼎鼎大名的 BERT 想必都有所耳闻了，在18年刚刚问世的时候在11个不同的 NLP 任务上取得了 state-of-the-art 的成绩，包括 QA，NER，Test Classification 等任务，那是不可谓不风光啊。即便到了现在，虽然 XLNet，T5 都出来了，BERT 的重要性还是不可同日而语。就把 BERT 的主要贡献放在最前面来说，就当做是个结论了。

**贡献1：** BERT 在文章里证明了双向向量表征的重要性 （Bidirectional Representation）。自从 BERT 得到了这个结论，后面的模型，如 XLNet，ALBERT 都把双向作为一个默认配置。
**贡献2：** 奠定了 pre-train + fine-tune 两阶段建模的实践基础。虽然以前也有 pre-train 模型，如 word2vec，GPT，ELMo，但是没有哪个起到了 BERT 这样大的作用。在 NLP 领域，后来的研究也都 follow 了这样的两阶段设定。

### 1. Motivtion of BERT
这一小节里，首先来描述一下为什么要提出 BERT。BERT 的主要贡献之一是证明了双向向量表示的重要性。这里所说的双向是深度双向表示，我们慢慢来说。

![](https://github.com/tonywenuon/posters/blob/master/images/bert1/3models.png?raw=true)

见上图，三个模型的对比中我们可以看到。传统的语言模型（Language Model）是从左向右（Left-to-Right）建模的。如图中的 OpenAI GPT，从第二层开始，每一个 token 的向量计算都是只用到了他左侧的前一层向量。这种模型的缺点是，当前 token 看不到他右边的 token 的信息，因此某种程度上，这种从左到右建模的模型天然的没有充分利用信息。接下来看图中的 ELMo，ELMo 模型采用的双向 LSTM，然后再最后一层把左右 LSTM concatenate 到一起，形成每个 token 的最终向量表示。但是 ELMo 是左侧，右侧向量表示独立训练，最后才 concatenate 到一起的。左侧和右侧向量表示在计算的时候并没有交互，因此他的语义表示比较浅层。那么这个时候就提出了一个问题：是否可以提出一个模型，在训练阶段计算某个 token 的向量表示的时候，既能利用上它左侧 token 的信息，又能


---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg2NjE2MTU3Myw0MzI5MjY1NzIsMTM5NT
Q5OTM3LC02NjgxNTIyOTAsLTEwMDU5NzY4OV19
-->