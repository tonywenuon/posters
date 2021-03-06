
# 【BERT 系列 1】之 BERT 本尊

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是 BERT？BERT 的创作动机是什么？
* BERT 最重要的贡献是什么？
* 如何用 BERT 来实现几个下游 NLP 任务？
* BERT 用到了哪些数据集？以及数据集的简介和地址是什么？
* 如何实现 BERT 代码？

#### 重要文章
* <span id = "paper1">Paper 1</span>:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

鼎鼎大名的 BERT 想必都有所耳闻了，在18年刚刚问世的时候在11个不同的 NLP 任务上取得了 state-of-the-art 的成绩，包括 QA，NER，Test Classification 等任务，那是不可谓不风光啊。即便到了现在，虽然 XLNet，T5 都出来了，BERT 的重要性还是不可同日而语。就把 BERT 的主要贡献放在最前面来说，就当做是个结论了。BERT 是一个层次 Transformer 模型，关于 Transformer 的介绍，请参见 [【重要系列 2】之 Transformer](https://zhuanlan.zhihu.com/p/93488997)。

**贡献1：** BERT 在文章里证明了双向向量表征的重要性 （Bidirectional Representation）。自从 BERT 得到了这个结论，后面的模型，如 XLNet，ALBERT 都把双向作为一个默认配置。
**贡献2：** 奠定了 pre-training + fine-tune 两阶段建模的实践基础。虽然以前也有 pre-train 模型，如 word2vec，GPT，ELMo，但是没有哪个起到了 BERT 这样大的作用。在 NLP 领域，后来的研究也都 follow 了这样的两阶段设定。如下图的两阶段，在 pre-training 之后，对下游任务如 SQuAD，NER，MNLI 进行 Fine-Tuning。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/bert1/2stages.png?raw=true)<br><br>

### 1. Motivation of BERT
这一小节里，首先来描述一下为什么要提出 BERT。BERT 的主要贡献之一是证明了双向向量表示的重要性。这里所说的双向是深度双向表示，我们慢慢来说。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/bert1/3models.png?raw=true)<br><br>

见上图，三个模型的对比中我们可以看到。传统的语言模型（Language Model）是从左向右（Left-to-Right）建模的。如图中的 OpenAI GPT，从第二层开始，每一个 token 的向量计算都是只用到了他左侧的前一层向量。这种模型的缺点是，当前 token 看不到他右边的 token 的信息，因此某种程度上，这种从左到右建模的模型天然的没有充分利用信息。接下来看图中的 ELMo，ELMo 模型采用的双向 LSTM，然后再最后一层把左右 LSTM concatenate 到一起，形成每个 token 的最终向量表示。但是 ELMo 是左侧，右侧向量表示独立训练，最后才 concatenate 到一起的。左侧和右侧向量表示在计算的时候并没有交互，因此他的语义表示比较浅层。那么这个时候就提出了一个问题：是否可以提出一个模型，在训练阶段计算某个 token 的向量表示的时候，既能利用上它左侧 tokens 的信息，又能用上右侧 tokens 的信息。当然啦，BERT 就是这么来的啦。

<span id = "bert_input">**BERT 的输入**</span> 

> **Input** = `[CLS]` the man went to `[MASK]` store `[SEP]` he bought a gallon `[MASK]` milk `[SEP]`
> **Label** = IsNext
> **Input** = `[CLS]` the man `[MASK]` to the store `[SEP]` penguin `[MASK]` are flight ##less birds `[SEP]`
> **Label** = NotNext

先给出 BERT 的输入，它的输入既可以是单个句子，也可以是句子对（如问答对）。两个句子中间用 `[SEP]` 标记来隔开。所以的输入的第一个 token 总是 `[CLS]` 这个标记的最终向量表示用来表征整个句子的向量，并且可以用做分类问题。除了额外加的 `[CLS]`，`[SEP]` ，`[MASK]` 标记，BERT 还引入了 Segment Embeddings 和 Position Embeddings。见下图，Position Embeddings 和 Transformer 中的 Position Embeddings 并无区别。Segment Embeddings 则是用来区分两个句子的向量，如果你的下游任务的输入只有一个句子，那么这里就只有 ![](https://latex.codecogs.com/gif.latex?E_A)。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/bert1/bert_embedding.png?raw=true)<br><br>

### 2. BERT pre-training 之 Masked Language Model（MLM）
说的容易做的难。如果一个 token 在训练的时候就都包含了左右的信息（当然了，也包含自己的），那岂不就相当于知道自己的信息还预测自己，如果这都可以，那还用那么多模型干啥。BERT 首先做的就是在原始的 sequence 的tokens 里面，随机的选择 15% 来屏蔽掉，即 mask 掉。然后这些被 mask 掉的 tokens 用来做预测的 targets，即 BERT 的目标之一就是预测这些被 mask 了的 tokens。还是看上面给出的这个例子。


> **Input** = `[CLS]` the man went to `[MASK]` store `[SEP]` he bought a gallon `[MASK]` milk `[SEP]`


这个例子中的 `[MASK]` 来替代原始的 token。好啦，这样我们预测 `[MASK]` 的 token 就可以啦。但是关于这个 `[MASK]` 还有个问题，大家先想一下如果是你来考虑，后面会有什么问题呢？我们继续，在做 pre-training 的时候是没有问题的，可以很好的预测 token。但是当把 BERT 用到下游任务的时候，问题就来了，在下游任务，是没有 `[MASK]` 标记的。你不能要求下游任务的训练集，测试集都有 `[MASK]` 标记，那这个模型就不 general 了。BERT 是这样做的，在刚才的 15% 的被 mask 了的 tokens 中，80% 的保持 `[MASK]` 状态，10% 用随机 sample 一个 token 来替代 `[MASK]`，10% 的保持原 token 不变。这样就允许模型看到一些非 `[MASK]` 的 token，虽然也看到了被 mask 的token 自身。不过也就只有 15%*10% = 1.5% 的样本，对整体模型效果的影响不大。至此，BERT 就构建完了。谢谢大家的关照，可以关闭本帖子了。哈哈，皮一下很开心。我们继续吧。


### 3. BERT pre-training 之 Next Sequence Prediction（NSP）
实际上，确实上面就是 BERT 最核心的部分。那么为什么还要有 NSP 呢。这是因为，很多下游任务，例如问答，自然语言推理等任务，都是依赖于两个句子的关系的。而这一需求，从上面的 MLM 中并体现不出来。因此作者设计了这个 Next Sequence Prediction 任务。还记得上面的 `[CLS]` 标记吗，它的输出可以认为是整个 sequence 的向量，所以在这里，就用 `[CLS]` 的向量表示来做一个二分类任务，来判断输入中的 sequence B 是否是 sequence A 的下一个 sequence。[参见上面例子](#bert_input) ，输入里面的 `Label` 就是是否是下一个句子的 ground truth了。训练的时候呢，50% 的 sample 输入的 sequence B 部分是真实的下一句，即 `Label = IsNext`。另 50% 的输入 sequence B 是随机生成的，也就是 `Label = NotNext`。在最后的实验分析中也有对比，增加 NSP 任务，对整体模型是有正向效果的。


### 4. BERT fine-tuning
#### 4.1 GLUE 实验
首先我们用 ![](https://latex.codecogs.com/gif.latex?C\in\mathbb{R}^H) 来表示 `[CLS]` 的最终向量表示。那么对于 GLUE 任务，在 Fine-Tuning 阶段，只需要引入一个分类层就可以了，这个层里包含可训练参数 ![](https://latex.codecogs.com/gif.latex?W\in\mathbb{R}^{K\timesH})。这里 K 是最终预测的 label 的个数，H 是向量维度。接下来，最终的预测类别就可以用 ![](https://latex.codecogs.com/gif.latex?log(softmax(CW^T))) 来表示。如下图所示，你所需要的就是用上面的公式和 C 的向量表示就可以了。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/bert1/mnli.png?raw=true)<br><br>

#### 4.2 SQuAD v1.1
Stanford Question Answering Dataset (SQuAD v1.1) 包含了 100k 的 QA 对。其目的是给定一个问题，然后在所给的段落中查找出这个问题所对应的答案。其实很类似一个答案抽取的问题。具体的做法是，把问题放到 segment A 的位置，把包含答案的段落放在 segment B 的位置来 Fine-Tuning。接下来我们引入两个向量。![](https://latex.codecogs.com/gif.latex?S\in\mathbb{R}^H) 来表示答案起始位置判断向量，![](https://latex.codecogs.com/gif.latex?E\in\mathbb{R}^H) 来表示答案结束位置判断向量。然后判断一个词是否是起始词，就用该词 ![](https://latex.codecogs.com/gif.latex?T_i) 和 S 的内积来表示其起始概率，判断一个词是否是答案结束词，就用该词 ![](https://latex.codecogs.com/gif.latex?T_i) 和 E 的内积来表示其结束概率。后面在跟一个 softmax 就得到了每个词作为起始或者结束词的概率。最后，一个答案片段的最终分数用 ![](https://latex.codecogs.com/gif.latex?S\cdotT_i+E\cdotT_j) 来表示，其中 i 和 j 是不同的 token 位置。取得到最大分数的文字片段作为最终的答案。见下图来形象化的理解上面的解释。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/bert1/squad.png?raw=true)<br><br>

#### 4.3 SQuAD v2.0
SQuAD v2.0 跟 SQuAD v1.1 的区别是在 v2.0 中，给定的段落并不一定包含问题的答案，其实这样的设定也跟实际生活中更加贴近。解决这个问题，我们可以在上面的解决方案上进行扩展。还记得 `[CLS]` 标记吧（感觉这个标记在 BERT 中是万能的，啥都能用它）。在预测答案的起始终止位置的时候，我们增加一个 `null` 答案来表示该段落中没有答案，概率用 ![](https://latex.codecogs.com/gif.latex?s_{null}) 来表示， ![](https://latex.codecogs.com/gif.latex?s_{null}=S\cdotC+E\cdotC) 。根据 4.2 的方法，我们还能够得到一个文字片段的概率，用 ![](https://latex.codecogs.com/gif.latex?s_{i,j}) 来表示。那么只有当 ![](https://latex.codecogs.com/gif.latex?s_{i,j}-s_{null}>\tau) 的时候，才使用从段落里抽取的答案，否则判断为该段落里没有答案。而这个 ![](https://latex.codecogs.com/gif.latex?\tau) 呢，则是根据在 dev set 上最大化 F1 计算出来的。

#### 4.4  SWAG
Situations With Adversarial Generations（SWAG）数据集包含了 113k 个句子对。其目的是，给定一个句子 A 和 4 个其他句子，来判断 4 个句子中，哪个句子是句子 A 最合适的后续句子。为了解决这个问题，首先把句子 A 和其余 4 个句子 concatenate 起来作为输入。判断的方法跟 4.1 中的 GLUE 一样。只需要引入另外一个层，这个层里包含一个参数向量 ![](https://latex.codecogs.com/gif.latex?W\in\mathbb{R}^{K\timesH})，计算向量 W 和 C 的内积，在跟着一个 softmax，用计算出来的概率来表达当前这个句子是否是句子 A 的后续句子。

以上就是 GLUE，SQuAD 和 SWAG 如何在 BERT pre-training 的基础上进行 Fine-Tuning 的方法。根据文章中的描述，在 Fine-Tuning 的过程，只需要数个小时就可以训练完成了，因为只需要几个 epoch 就可以收敛。

### 5. GLUE 数据集

General Language Understanding Evaluation (GLUE) 是一个自然语言理解的 benchmark。它包含了 9 个不同的自然语言处理任务，数据可以在[这里下载](https://gluebenchmark.com/tasks)到。下面我会分开介绍一下每个任务都是什么。整体的 [leaderboard ](https://gluebenchmark.com/leaderboard/) 请点击链接查看（目前已被 Google T5 屠榜）。里面的英文名字我就不翻译了，我也还没找到 commonly accepted 的中文翻译名。如果谁知道每一个任务的中文名，烦请指教。

**（1）MNLI（392k）** 
全称 Multi-Genre Natural Language Inference。MNLI 是一个蕴含关系分类的任务，它是一个大语料集，BERT 中表述用了 392k 的数据。它的任务是，给定一个句子对，目的是预测第二个句子跟第一个句子是否是：包含，互斥还是中立关系。
**（2）QQP（363k）** 
Quora Question Pairs 是一个二分类任务，其目的是判别在 Quora 上的两个问题是否在语义上是相同的。
**（3）QNLI（108k）** 
Question Natural Language Inference，是 Standford Question Answering 数据集的另一个版本，与原始版本不同，这里转换成了一个二分类任务。正样本是 （question，sentence）这个句子对中包含了正确的答案，负样本是句子对中不包含答案。
**（4）SST-2（67k）** 
Standford Sentiment Treebank 是一个单句子的二分类任务，他包含了关于影评的句子。人工的来标注他们对该电影的评价。
**（5）CoLA（8.5k）** 
Corpus of Linguistic Acceptability 也是一个单句子的二分类任务。其目标是判断一个英文句子是否在语言上是“可接受的”。
**（6）STS-B（5.7k）** 
Semantic Textual Similarity Benchmark 是句子对集合，他是从新闻的标题和其他来源聚合而成。他的标注是从 1 分到 5 分的标注，来表示给定的两个句子在语义上的相似度有多大。
**（7）MRPC（3.5k）** 
Microsoft Research Paraphrase Corpus 也是句子对集合。他是自动的从互联网新闻中抽取出来并且由人工来标注给定的句子对是否语义一致。
**（8）RTE（2.5k）** 
Recognizing Textual Entailment 是一个和 MNLI 很类似的数据集但是他的数据规模比较小。
**（9）WNLI** 
Winograd NLI 是一个小的自然语言推理数据集。BERT 中并没有 report 这个结果，因为 WNLI 的 train 和 dev set 的设置有问题，并且 test set 的分布和 train set 的分布不同。

### 6. 如何实现 BERT 代码？
笔者没有自己实现 BERT 的代码，给出几个源码连接，大家感兴趣的自己去学习吧。

-   **Google原版bert**:  [https://github.com/google-research/bert](https://github.com/google-research/bert)
- **CyberZHG 大佬版**：[https://github.com/CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)
-  **bojone 版**：[https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)

---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMjYyMTk5MTgsLTMzMTk1MTUwMiw1Nj
c0NzEzOTksLTIxMjMzNzM4MzAsMjA1NzE1OTU3MywtOTM3Nzg1
OTM4LDIyNjkyMDEyMywxNTc1MjA5Mjk2LDEwNDk1ODkzMDMsND
MyOTI2NTcyLDEzOTU0OTkzNywtNjY4MTUyMjkwLC0xMDA1OTc2
ODldfQ==
-->
