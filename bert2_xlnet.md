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

之所以把 XLNet 归到 BERT 系列是因为 XLNet 的逻辑是提升 BERT 模型天然的短板，弥补了 BERT 中的两个缺陷。加上在他们之后发布的文章很多也会拿他们俩来比较，我在这里也把他们分到同一个系列中。原文章里涉及到比较多的公式，我这里能省就省了。对 BERT 不了解的童鞋们，可以先读一下 [【BERT 系列 1】之 BERT 本尊](https://zhuanlan.zhihu.com/p/94513051)

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
**BERT 问题1：预测独立性假设**，在预测 [MASK] 位置的词的时候，每个 [MASK] 都是独立地预测，比如：

> 原句子：深圳是个大都市
> Mask 句子：[MASK] [MASK] 是个大都市

可以看到，在预测的时候，这两个 [MASK] 相当于是这样预测的，在给定 `是个大都市` 的条件下，预测 `深` 和 `圳` 的概率分别是：p(深 | 是个大都市) 和 p(圳 | 是个大都市)。而这显然是有缺陷的，没有考虑到 `深圳` 这两个词的关系（当然了，这里不考虑汉语的词表，只考虑单个字）。
**BERT 问题2：训练-预测不一致**，在训练好 BERT 与训练模型以后，在下游任务进行 fine-tuning。但是下游任务是没有 [MASK] 标记的。这就引起了训练-预测不一致的问题，虽然说在 BERT 训练好了以后，基本上能够生成出每个词的语义向量，但是毕竟有 [MASK] 的存在，这不是一个自然的预测下游任务的方式。

**语言模型问题：单向模型**，说完了 BERT，我们再来说语言模型，虽然语言模型没有上述问题，但是其最大的短板是非双向可达，双向可达是指在计算一个字的语义的时候，这个语义应该既考虑到前面的字的信息，又考虑到后面的字的信息。BERT 已经明确验证过双向语义模型对于语义表达的重要性。但是在语言模型中，只有单向，最常见的是从左向右建模，即右边的字只能看到其左边的字的信息。语言模型的优点是符合自然预测的习惯即根据前面的信息，来生成后面的文字。

BERT 和语言模型有各自的优缺点，那么有没有一种方式能结合两者的优点呢？即结合 BERT 的双向语义建模和语言模型从左至右预测的特点。当然有啦，就是 XLNet 咯。

### 3. XLNet 之排列语言模型（Permutation Language Model）
为了解决上面的问题，XLNet 提出了排列语言模型。其含义是说，每个序列输入进来以后，他都会有各种排列组合，比如一个序列一共有 T 个词，那么他的排列组合就有 T! 个。这个时候，依然遵循语言模型的规则，当计算某个特定位置 t 的时候，在所有的排列组合中，t 能看到所有的信息（包括前面和后面），因为大于 t 的位置的字都会出现在小于 t 的位置上。这么说比较抽象，拿个例子来说，括号里的表示每个字对应的原序列的位置。

> 原序列： 我  爱  中  国 （1,2,3,4）
> 排列1：我  爱  国  中（1,2,4,3）
> 排列2：我  中  爱  国（1,3,2,4）
> 排列3：我  中  国  爱（1,3,4,2）
> 排列4：中  我  爱  国（3,1,2,4）
> ......

这里不把全部的排列都列出来，只介绍重点含义就好了。看这个例子，我们拿 `中`（序列号是3）为例。由于训练方式还是用语言模型的方式来训练，站在某个字上，他只能看到其左边字的信息，所以在原序列中，`中` 可以看到 `我` 和 `爱` 的信息，但是看不到 `国` 的信息。但是在排列 1 中，`中` 却可以看到所有其他字的信息，因为 1,2,4 都排在 3 的前面。同样的道理，对于 `我` 在原序列中，看不到后面的任何信息，但是通过排列以后，1 就可以看到 2,3,4 的信息了，因为在排列 `2,3,4,1` 中，1 就可以看到所有的字（因为他们在 1 的左边）。好了，啰嗦了这么多，总结起来说就是，通过排列，每个字都可以看到其他所有字的信息，因为他们都有机会排列到其他字的左侧，这样语言模型就能够看到他们。接下来就是如何设计目标函数了，XLNet 的目标函数是，是的所有排列的生成概率都最大，即：

![](https://github.com/tonywenuon/posters/blob/master/images/bert2/objective.png?raw=true)

这里的 $Z_T$ 表示所有的排列，$x_{z<t}$ 指的是所有排列在 t 前面的 token。其实看到这个公式里，在括号部分就是语言模型，只是把这个语言模型应用到了所有的排列后的序列上，并且让所有排列序列的生成概率都最大。这样就是侧面的让一个词看到了其他所有词的信息，那么通过这样的方式，这个模型既遵循了语言模型从左向右建模的特点，又整合了双向语义建模的结构。这就是 XLNet 最核心的思想。接下来的问题就是如何实现这一思路。

### 4. 双流自注意力（Two-Stream Self-Attention）

介绍双流注意力之前要指出“两个不可以”。
**标准语言模型不可以**，整个 XLNet 框架都是用 Transformer 来实现的（有对 Transformer 不了解的，请参见 [【重要系列 2】之 Transformer](https://zhuanlan.zhihu.com/p/93488997)）。而 Transformer 在计算的时候，每个 token 的 vector 都要参与到其他 token 的计算中。这对于未排列的句子是没有问题的，但是对于排序语言模型就行不通了。还是拿 `1,2,3,4` 来举例子，假设现在我们有两个排序序列：（1）我爱中国（`1,2,3,4`）；（2）我爱国中（`1,2,4,3`）。前面两个位置都是 `1,2` 那么根据标准语言模型，第三个位置的概率是 $p(中 | 我爱)$，而问题是 $我爱$ 在这两个序列中是完全一样的，所以无法直接套用标准语言模型。为了解决这个问题，在 XLNet 中，作者把 token 和 token 的位置区分开（文章中叫做 re-parameterize）。即在（1）的例子中，$p(中 | 我爱; 3)$，在（2）中，$p(国 | 我爱; 4)$。这样引入了 target 位置信息到概率计算中，就解决了 *标准语言模型不可以* 的问题。为了后面好解释，这里还是给出一个正式的公式：$p(x_{z_t}=x | x_{z<t}) = p(x_{z<t}, z_t)$，其中 $z_t$ 是当前要预测的位置（注意是排序位置，不是绝对位置，拿（2）为例，4 在序列的第三个位置，即 t=3，那么这个 $z_t$ 是 4，而不是 3）。
**标准 Transformer 不可以**，在标准语言模型中，一个 token 要么参加其他 token 的计算，要么不参加，即非 0 即 1 的关系。这在排序语言模型，就是 XLNet 中就引出了下面的矛盾。（1）为了预测 $x_{z_t}$，根据上面的讨论，只用到 $z_t$（位置信息），而不是 $x_{z_t}$（token 内容信息）；（2）当预测 $x_{z_j}$，j>t 的时候，即预测 t 位置之后的 token 的时候，又需要用到 内容信息 $x_{z_t}$。标准 Transformer，$x_{z_t}$ 要么加入，要么不加入，此时他就无法使用了。这个时候才引出了双流注意力机制。

**双流注意力**
* **Content representation**，简写为 $h_{z_t}$，这个向量表征在计算的时候和标准 Transformer 一模一样，即每个 token 的内容加入其他 token 的计算。
* **Query representation**，简写为 $g_{z_t}$，而这个向量表征的计算，只用当前的位置信息计算，而不是内容。

![](https://github.com/tonywenuon/posters/blob/master/images/bert2/two_stream.png?raw=true)

可以看到，公式中一个是 $z<t$，一个是 $z \leqslant t$，还可以看到在计算 g 的时候，除了 $z_t$，其他的 token 都是用的 h，即内容向量表示来计算。更形象化的来理解，我们一起看下图。

![](https://github.com/tonywenuon/posters/blob/master/images/bert2/content.png?raw=true)

![](https://github.com/tonywenuon/posters/blob/master/images/bert2/query.png?raw=true)

图 (a) 表达的是和正常的 Transformer 一样，每个 h，即 token 内容都参与计算。图 (b) 表达的是，计算 g 的时候，当前的 token，只有其位置（图中是 $g_1^{(0)}$）参与向量的计算。

在具体实现的时候，并不是把输入序列直接进行排列，而是通过 attention mask 来在 Transformer 内部实现，输入保持原序列的样子。我们一起来看下图。

![](https://github.com/tonywenuon/posters/blob/master/images/bert2/content_query.png?raw=true)

可以忽略这张图的左侧，只关注右侧。右侧是 `3-2-4-1` 这个排列的 mask，我们一起来看上面的矩阵，即 content representation 的 mask 矩阵。每一行，每一列表示原序列中的位置，从 1 到 4。由于在排列 `3-2-4-1` 中，1 是在最后，因此在计算 content stream 的时候，他可以看到所有其他 token 的内容，因此在矩阵里，每一列都是 1，也就是标红的。那么第二行呢，可以看到在排列中，2 只能看到其前面的 3，所以第二行只有第 2 和第 3 列标红。以此类推，第三行只有第 3 列标红，第四行有 2,3,4 列标红。这都是依据排列中的顺序来定的。接下来我们看下面的 mask 矩阵，即 query stream。图中用虚线把对角线去掉了，也就是说在计算 query representation 的时候，每个 token 自身的内容是不参与计算的，所以直接把 content mask 矩阵的对角线去掉就可以了。

### 5. 段循环机制和相对位置编码

对应到原文，段循环机制叫做 segment recurrence mechanism；相对位置编码叫做 Relative Segment Encodings。这两个概念都是从 Transformer-XL 引入的，感兴趣的读者可以详细的阅读其文章，因为他们不是 XLNet 的原创，在这里我就简单介绍他们的功能。段循环机制是为了解决长句依赖问题的。因为在 BERT 中，句子在训练的时候是相对独立的，也就是每个 sample 并没有前面 sample 的信息。引入段循环机制后，每个句子在计算向量表示的时候还会加入上一个句子的信息，见下图，也就是图中的 mem，上个句子的每一层的 memory 都会被保存。这样赋予了 XLNet 更强的长句记忆的功能。

![](https://github.com/tonywenuon/posters/blob/master/images/bert2/content_split.png?raw=true)

而相对位置编码相对于 BERT 的绝对位置编码更加灵活。还记得在 BERT 中 segment A 和 segment B 的编码是强行将各自的位置 embedding 加到 word embedding 中。在 XLNet 中，位置编码只区分两个位置是否在同一个 segment 中，而不做绝对区分。具体来说，如果位置 i 和位置 j 来自于同一个 segment，那么 $s_{ij}=s_+$，否则 $s_{ij}=s_-$，这里的 $s_+$ 和 $s_-$ 是两个可训练参数向量。那么当位置 i 的token 参与到位置 j 的向量计算的时候，根据位置编码来计算一个 attention，$a_{ij}=(q_i+b)^Ts_{ij}$，这里的 $q_i$ 是 i 位置的向量表示。计算完以后再把这个 attention 加到正常的 attention 中，当做是位置的 attention。文章中也指出这样的相对位置编码的好处是更加灵活。一方面这提升了模型的泛化能力，另一方面想象下下游任务中如果训练的输入要求不是两个 segment 的拼接，而是三个以上拼接的时候，使用绝对位置编码就不可行了。


### 6. XLNet 的训练
双流注意力是 XLNet 实现中最不好理解的，理解了这部分其他部分就好办了。那么在训练的时候，输入采用了和 BERT 类似的 segment 方法。唯一的区别是 `[CLS]` 的位置。

> **BERT Input** = `[CLS]` the man went to `[MASK]` store `[SEP]` he bought a gallon `[MASK]` milk `[SEP]`
> **XLNet Input** = the man went to `[MASK]` store `[SEP]` he bought a gallon `[MASK]` milk `[SEP]` `[CLS]`

可以看到，BERT 把`[CLS]` 作为第一个 token，而 XLNet 作为最后一个 token。同时对于训练目标，BERT 是预测被 mask 掉的词。而 XLNet 则是在每个输入的句子中间选择一个位置 c，来进行预测 c 以后的部分句子。实现起来是，设定一个超参数 K，那么 1/K 的词用来预测，其他的词用来训练。K 越大，则越多的上下文信息可用，也就越精确。文章中也有对 K 的的设定实验。

### 7. XLNet 和 BERT 的独立性假设分析

首先，XLNet 把 Next Sentence Prediction 去掉了，因为实验结果显示，这个 NSP 任务并不会提高模型效果，甚至还有副作用。当然了，这是从结果反推的结论。这里我想讨论的是在训练过程中两者的异同，参考了 @张俊林 大神的文章 [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)，感兴趣的小伙伴自行关注。

在原文章中，作者指出了 BERT 的独立性假设，即预测 [MASK] 位置的词的时候，各个 [MASK] 是独立的，而实际情况下，他们可能并不独立，一个 [MASK] 可能依赖于另一个 [MASK]，如假设 “北京” 被 mask 掉，那么预测 “京” 的时候，是要依赖于 “北” 的。而 XLNet 可以解决这个问题。这个问题从表面上看是这样的，XLNet 通过排列，能够使得每个词都看到其他词，也就达到了非独立性假设的目的。但是试想，这是靠着排列来实现的，也就是说数据冗余，每一条数据被排列成 n 份，通过这样的数据冗余，总能使得各个词都有依赖关系。那么 BERT 呢？他真的做不到非独立吗？其实不然，BERT 在训练过程中每个 sample 只有一份，被 [MASK] 掉的词在当前的 sample 中确实相互预测时是独立的。那么这里就有个潜在的待验证的问题，通过大量数据，甚至是海量数据，总有其他的 sample 在预测的时候能把这个 sample 的两个词给关联起来。假设其他的句子中 “京” 被 mask 了，而 “北” 没有被 mask，那么就有了这样的依赖关系。所以在大量数据训练的情况下，可以认为 BERT 不存在独立性假设。另外，他们俩不能直接对比独立性假设的另一个原因是数据冗余问题（XLNet 通过排列存在数据冗余，BERT 不存在），这里做个大胆的设想，假设 BERT 在训练的时候，对于每一个 sample 不是只做一次 15% 的 mask，而是在随机 mask 以后，对所有被 mask 了的词，挨个的 unmaks 即去掉其 mask。这样相当于每个 sample 又生成了 n 个可训练 sample。通过这样的设定，被 mask 了的词，总能看到其他被 mask 的词的信息。这样也解决了独立性假设的问题。所以综上，依赖于大量数据或者数据冗余，BERT 也都不存在独立性假设的问题。基于这样的分析，可以认为 BERT 从模型设计的出发点上并不逊于 XLNet，也许这也是为什么后面有 RoBERTa 和 ALBERT 工作出现的原因吧。尤其 RoBERTa 工作，指出了如果对 BERT 的设计进行合理的配置，BERT 是并不比目前任何与训练模型差的，当然包含 XLNet。RoBERTa 和 ALBERT 会作为 BERT 系列的 3 和 4 来介绍。

但正如 @张俊林 大神所说，毕竟一个 AR 任务，一个是 AE 任务，在生成式设定上，XLNet 的设计选择优于 BERT，毕竟他遵从了从左到右建模及预测的程式，因此在生成式任务的表现，理论上 XLNet 应优于 BERT。

对于上面所提到的 RoBERTa ，感兴趣的小伙伴请关注后续【BERT 系列 3 】之 RoBERTa，精化 BERT 模型。

---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMjIxNjQ0NTcsMTA5NDg2MTc5NiwtMT
EzNDQ4NDY2Miw3NTgzNjY2NywtMjEyMTY2NDIzMywtMTI5NDky
ODkwOCw3MzUwMTc2NTAsLTE3MTg3Nzg2MDcsMjA3MDkzMjA4NC
wtMTMzOTU3MDM5MywxNjg3ODY4NTgzLC0xNjk1MTA5NzQwLC0x
MDM4MTg5MjY4LC05NTk5MTI0OCwtODQ0MDczNTIsMzA2NzAyOD
c5LC0xMzgzOTIxMzkxLC01NTM4ODA4MzUsLTE3MDg4NDU3ODZd
fQ==
-->