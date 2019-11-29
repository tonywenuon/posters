
# 【BERT 系列 1】之 BERT 本尊

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是 BERT？BERT 的创作动机是什么？
* BERT 最重要的贡献是什么？
* 如何用 BERT 来实现几个下游 NLP 任务？
* BERT 的局限性在哪里？
* BERT 用到了哪些数据集？以及数据集的简介和地址是什么？

#### 重要文章
* <span id = "paper1">Paper 1</span>:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

鼎鼎大名的 BERT 想必都有所耳闻了，在18年刚刚问世的时候在11个不同的 NLP 任务上取得了 state-of-the-art 的成绩，包括 QA，NER，Test Classification 等任务，那是不可谓不风光啊。即便到了现在，虽然 XLNet，T5 都出来了，BERT 的重要性还是不可同日而语。就把 BERT 的主要贡献放在最前面来说，就当做是个结论了。BERT 是一个层次 Transformer 模型，关于 Transformer 的介绍，请参见 [【重要系列 2】之 Transformer](https://zhuanlan.zhihu.com/p/93488997)。

**贡献1：** BERT 在文章里证明了双向向量表征的重要性 （Bidirectional Representation）。自从 BERT 得到了这个结论，后面的模型，如 XLNet，ALBERT 都把双向作为一个默认配置。
**贡献2：** 奠定了 pre-training + fine-tune 两阶段建模的实践基础。虽然以前也有 pre-train 模型，如 word2vec，GPT，ELMo，但是没有哪个起到了 BERT 这样大的作用。在 NLP 领域，后来的研究也都 follow 了这样的两阶段设定。

### 1. Motivtion of BERT
这一小节里，首先来描述一下为什么要提出 BERT。BERT 的主要贡献之一是证明了双向向量表示的重要性。这里所说的双向是深度双向表示，我们慢慢来说。

![](https://github.com/tonywenuon/posters/blob/master/images/bert1/3models.png?raw=true)

见上图，三个模型的对比中我们可以看到。传统的语言模型（Language Model）是从左向右（Left-to-Right）建模的。如图中的 OpenAI GPT，从第二层开始，每一个 token 的向量计算都是只用到了他左侧的前一层向量。这种模型的缺点是，当前 token 看不到他右边的 token 的信息，因此某种程度上，这种从左到右建模的模型天然的没有充分利用信息。接下来看图中的 ELMo，ELMo 模型采用的双向 LSTM，然后再最后一层把左右 LSTM concatenate 到一起，形成每个 token 的最终向量表示。但是 ELMo 是左侧，右侧向量表示独立训练，最后才 concatenate 到一起的。左侧和右侧向量表示在计算的时候并没有交互，因此他的语义表示比较浅层。那么这个时候就提出了一个问题：是否可以提出一个模型，在训练阶段计算某个 token 的向量表示的时候，既能利用上它左侧 tokens 的信息，又能用上右侧 tokens 的信息。当然啦，BERT 就是这么来的啦。

<span id = "bert_input">**BERT 的输入**</span> 

> **Input** = `[CLS]` the man went to `[MASK]` store `[SEP]` he bought a gallon `[MASK]` milk `[SEP]`
> **Label** = IsNext
> **Input** = `[CLS]` the man `[MASK]` to the store `[SEP]` penguin `[MASK]` are flight ##less birds `[SEP]`
> **Label** = NotNext

先给出 BERT 的输入，它的输入既可以是单个句子，也可以是句子对（如问答对）。两个句子中间用 `[SEP]` 标记来隔开。所以的输入的第一个 token 总是 `[CLS]` 这个标记的最终向量表示用来表征整个句子的向量，并且可以用做分类问题。除了额外加的 `[CLS]`，`[SEP]` ，`[MASK]` 标记，BERT 还引入了 Segment Embeddings 和 Position Embeddings。见下图，Position Embeddings 和 Transformer 中的 Position Embeddings 并无区别。Segment Embeddings 则是用来区分两个句子的向量，如果你的下游任务的输入只有一个句子，那么这里就只有 $E_A$。

![](https://github.com/tonywenuon/posters/blob/master/images/bert1/bert_embedding.png?raw=true)

### 2. BERT pre-training 之 Masked Language Model（MLM）
说的容易做的难。如果一个 token 在训练的时候就都包含了左右的信息（当然了，也包含自己的），那岂不就相当于知道自己的信息还预测自己，如果这都可以，那还用那么多模型干啥。BERT 首先做的就是在原始的 sequence 的tokens 里面，随机的选择 15% 来屏蔽掉，即 mask 掉。然后这些被 mask 掉的 tokens 用来做预测的 targets，即 BERT 的目标之一就是预测这些被 mask 了的 tokens。还是看上面给出的这个例子。


> **Input** = `[CLS]` the man went to `[MASK]` store `[SEP]` he bought a gallon `[MASK]` milk `[SEP]`


这个例子中的 `[MASK]` 来替代原始的 token。好啦，这样我们预测 `[MASK]` 的 token 就可以啦。但是关于这个 `[MASK]` 还有个问题，大家先想一下如果是你来考虑，后面会有什么问题呢？我们继续，在做 pre-training 的时候是没有问题的，可以很好的预测 token。但是当把 BERT 用到下游任务的时候，问题就来了，在下游任务，是没有 `[MASK]` 标记的。你不能要求下游任务的训练集，测试集都有 `[MASK]` 标记，那这个模型就不 general 了。BERT 是这样做的，在刚才的 15% 的被 mask 了的 tokens 中，80% 的保持 `[MASK]` 状态，10% 用随机 sample 一个 token 来替代 `[MASK]`，10% 的保持原 token 不变。这样就允许模型看到一些非 `[MASK]` 的 token，虽然也看到了被 mask 的token 自身。不过也就只有 15%*10% = 1.5% 的样本，对整体模型效果的影响不大。至此，BERT 就构建完了。谢谢大家的关照，可以关闭本帖子了。哈哈，皮一下很开心。我们继续吧。


### 3. BERT pre-training 之 Next Sequence Prediction（NSP）
实际上，确实上面就是 BERT 最核心的部分。那么为什么还要有 NSP 呢。这是因为，很多下游任务，例如问答，自然语言推理等任务，都是依赖于两个句子的关系的。而这一需求，从上面的 MLM 中并体现不出来。因此作者设计了这个 Next Sequence Prediction 任务。还记得上面的 `[CLS]` 标记吗，它的输出可以认为是整个 sequence 的向量，所以在这里，就用 `[CLS]` 的向量表示来做一个二分类任务，来判断输入中的 sequence B 是否是 sequence A 的下一个 sequence。[参见上面例子](#bert_input) ，输入里面的 `Label` 就是是否是下一个句子的 ground truth了。训练的时候呢，50% 的 sample 输入的 sequence B 部分是真实的下一句，即 `Label = IsNext`。另 50% 的输入 sequence B 是随机生成的，也就是 `Label = NotNext`。在最后的实验分析中也有对比，增加 NSP 任务，对整体模型是有正向效果的。


### 4. BERT fine-tune






---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTU3NTIwOTI5NiwxMDQ5NTg5MzAzLDQzMj
kyNjU3MiwxMzk1NDk5MzcsLTY2ODE1MjI5MCwtMTAwNTk3Njg5
XX0=
-->