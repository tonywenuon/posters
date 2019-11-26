
# 【重要系列 2】之 Transformer

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是 Transformer？
* Self-Attention 和 Multi-Head 机制是什么？
* 如何用 Keras 实现一个 Transformer 模型？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

本文不会面面俱到的把文章里所有的细节都介绍到，而是主要讲解最重要的几个部分。首先我们就从研究动机开始。
**Motivation:** 在 Transformer 被提出之间，基本上 RNN 架构统治了 sequence-to-sequence 模型。尤其在 Attention 机制被提出后，Seq2Seq with Attention 得到了广泛的应用（详细原理可参见[https://zhuanlan.zhihu.com/p/87961308](https://zhuanlan.zhihu.com/p/87961308)）。虽然 LSTM 或者 GRU 有记忆能力，一定程度上能够获取长距离记忆，但是一旦距离过长，那么他的作用就有限了。举个例子来说，比如原句子为 “The animal didn't cross the road because it was too tired.”，在这个句子里 `it` 指代的就是 `animal`。如果用 LSTM 来计算，那么 `animal` 的语义信息，要通过 5 次计算（计算5个词 didn't cross the road because）后才能到达 `it`，那么这个时候 `animal` 的信息还剩多少呢？这个不好说，但是可以肯定的是中间经过了 5 次的遗忘门（forget gate）会有信息损耗。那么有没有一种方式，能够让 `it` 和 `animal` 直接进行交互，不通过中间那么多层呢？当然有，这就该 Transformer 出马了。

## 1. 什么是 Transformer？

Transformer，变形金刚，我觉得这个翻译还挺有意思的，哈哈。言归正传，Transformer 实际上是一种架构，从架构上并体现不出来他为什么起名叫 Transformer。在 Transformer 中最重要的两个结构是 Self-Attention 和 Multi-Head。下面我们依据架构图逐一介绍。

![](https://github.com/tonywenuon/posters/blob/master/images/important2/transformer.png?raw=true)

如图左边部分是 Transformer 的 Encoder，右边是 Decoder。Encoder 主要有两层：Multi-Head Attention 和一个前向网络。其中的 `Add & Norm` 是 Residual Connection 和 Layer  Normalization。这里，前向网络，Residual Connection 和 Layer  Normalization 并不是本文的重点，我会在介绍完 Self-Attention 和 Multi-Head 之后再介绍一下他们的作用。

### 1.1 Self-Attention
还是拿这个例子来说，“The animal didn't cross the road because it was too tired.”。前面说到，当计算 `it` 的向量表示的时候，在 LSTM 里，会一步一步的从 `animal` 计算到 `it`，来得到 `it` 的向量表示。而 Self-Attention 呢，则是直接计算了 `it` 与其他所有的词的 Attention，把这个 Attention 当做 Weight 来对其他词的向量表示加权求和，最终得到 `it` 的向量表示。还是上图好理解，见下图。

![](https://github.com/tonywenuon/posters/blob/master/images/important2/self_attention.png?raw=true)

图中右边表示当计算 `it` 的 vector 的时候，会计算和左侧每一个词的相似度。颜色越深表示相似度越高。很明显，可以看到 `it` 和 `animal` 的相似度是最高的。然后每个词的相似度乘以其对应的向量表示，加到一起，就形成了 `it` 的 vector。

这段文字阐述呢，对应于原文章的公式：
![](https://github.com/tonywenuon/posters/blob/master/images/important2/sa_equation.png?raw=true)
公式中的 $\sqrt d_k$ 是尺度因子。根据原文的阐述，之所以除以这个因子，是因为当 $d_k$ 很小的时候，有没有这个因子区别不大。但是如果这个值很大，那么在计算 QK 点乘的时候会产生比较大的值，进而导致产生较小的梯度，影响训练。因此在这里应用了这个尺度因子。有没有觉得很简单。

### 1.2 Multi-Head
Multi-Head 也很简单。不要被 Head 所迷惑，此 “头” 非彼 “头”。这里的 Head 其实就是指把一个完整的向量分成几段。分成几段就是几 “Head”。例如，使用的向量长度是 512 （也就是 $d_k$）。这是说对于每一个词用 512 维来表示。那么当 head = 8 的时候，就是说把这 512 分成 8 份，每份的长度是 512/8 = 64。

![](https://github.com/tonywenuon/posters/blob/master/images/important2/multi_head.png?raw=true)

图中显示的就是 8 head 的结构。然后再每一个 64 维上进行 self-attention 的计算，这就是 Transformer 的第一层的设定了。也就是架构图中的 `Multi-Head Attention`。作者呢是假定，不同的 head 能够获取到这个词的不同维度的信息，所以多头会优于单头的设定。当然了，这一点假设在原文中并没有明确的验证。近期在 EMNLP 2019 会议中有人专门对多头进行了研究，指明其实在 Transformer Encoder 的多头设定中，这个多头是有很多冗余信息的，头多了反倒会对结果有负面的影响。至于到底设置成多少头呢，这个大家自己做实验的时候根据自己的情况来设定吧。哦，忘记了，在多头 vector 计算完 self-attention 后，怎么合并多头呢？文章里给了很简单的方法，直接 concatenate 起来就好了。
![](https://github.com/tonywenuon/posters/blob/master/images/important2/concate.png?raw=true)

### 1.3 Position Encoding

位置编码其实是个很好理解的问题。依然，拿 “The animal didn't cross the road because it was too tired.” 来举例子。假设没有位置编码的话。那么刚才这句话和 “It was too tired because the animal didn't cross the road.” 在计算完 Self-Attention 之后，没有任何区别。`it` 还是和每一个词计算了 Attention 然后加权求和。但直觉上是不对的。这两句话根本不一样，得到的语义信息不应该一样。再极端一点的例子，把这些词随机打算顺序，那么他们计算的结果还是都一样，那就更不对了。这就引出了 Position Encoding。原文中给出了两种方法，并指出两种方法经过实验结果的验证并没有显著区别。

* 基于三角函数的位置编码
* 基于自动训练的位置编码向量 （trainable vector）

自动训练的 position encoding 就不多说了，把每个词的 embedding 加个位置向量就 ok 了。基于三角函数的则是：

![](https://github.com/tonywenuon/posters/blob/master/images/important2/pe.png?raw=true)

可以看到作者是对于奇数偶数位置分开设定，而且使用三角函数的好处是三角函数的周期函数，这样能够一定程度上体现出相对位置的信息，而不是只体现绝对位置。至此，最主要的结构都介绍完了。

### 1.4 Masks

感谢 @李敬泉 的建议，在这里我再补充讨论一下两种 mask。实际上在 Transformer 中只有一种 mask，就是在 Decoder 阶段使用的 mask。另一种mask 是我根据自己经验介绍点关于 `<pad>` 的mask。

#### Decoder mask
拿个例子来说，假设现在给的场景是智能对话。在训练集中有这样一组问答。

> Question: Where did you live in the last two years ?
> Answer: I used to live in the Los Angeles .

那么，在训练的时候，整个 Answer 都会被输入到 Decoder 中。那么问题就来了，当你想要生成 `used` 的时候，模型是不应该看到 `used` 这个词和这个词以后的所有词的，而只应该看到 `I`，根据 `I` 来生成下一个词，即`used`。如果模型可以看到 `used`，那就不用预测了不是。所以在计算 `used` 这个词的时候，就应该用个 mask，把后面所有词的 vector 给 mask 掉，即他们不参与生成 `used` 这个词时的计算。Answer 里一共有 9 个词（包含标点），那么 mask 就是 `[1 0 0 0 0 0 0 0 0]`，在 self-attention 加权求和的时候，再乘以这个 mask 向量，就相当于过滤掉了 `used` 及其以后所有的词。

#### PAD mask
还是拿上面的 QA 例子来介绍。通常来讲，输入到模型的数据都要数据对齐。所谓数据对齐是说所有的数据都要保持同样长度。例如我们规定 Question 的最大长度是 20。那么那些长度大于 20 的 sequence 要被截断，长度小于 20 的要用一个特殊标记来对齐长度，例如用 “\PAD”。按上面的例子，Question 就要变成

> Question: Where did you live in the last two years ? \PAD \PAD \PAD \PAD \PAD \PAD \PAD \PAD \PAD \PAD

那么现在问题就来了。在你计算 self-attention 的时候，所有的 `\PAD` 也都会参与计算，但是 `\PAD` 又不是 sequence 里的内容。大量的 `\PAD` 如果不加以过滤，就相当于给向量表示引入了很多噪音。根据个人经验，这就会导致不论输入什么 Question，各个字符的向量表示都会比较相近（因为大家都是参合了 `\PAD` 的信息）。这样进而导致了 Transformer 的 Decoder 后的结果都很相似，也就是通常 paper 里所说的 generic，例如不论输入什么总是回复相同的 “I don't know.”。所以呢，这里要引入第二个 mask。对应于 sequence 长度，这个 PAD mask 就应该是 `[1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]`。在计算 self-attention 的时候，这就把 `\PAD` 给 mask 掉了。

### 1.5 Residual Connection 和 Layer Normalization
这两部分的设定都是 follow 前人的工作。Residual Connection 是说把优化目标由 $H(x) = f(x)$ 变成 $H(x) = f(x) + x$，这就是残差网络。他本身的出发点是从网络深度来的。理论上来说，越深的网络，其效果也是越好的。换句话说，深的网络不会比浅的网络效果差。但是实际情况却不是这样的，有时候由于网络太深导致难以训练，返到不如浅网络好。这一现象被称为**退化问题（degradation problem）**。残差网络就是解决这个问题的，残差网络越深在训练集的效果越好 (ref 1)。而 Layer Normalization 则是用来提高训练速度的。


## 2. 如何用 Keras 实现一个 Transformer 模型？

实现一个 Transformer 比 Seq2Seq 会复杂一些。这里只给出最关键部分的代码。给出的代码是如何计算 multi-head self-attention。

```python
   1 # shape (batch_size, head_number, sequence_length, d_k//head_number)
  2 q_shape = K.shape(q)
  3 v_shape = K.shape(v)
  4 k_t_shape = K.shape(k_transposed)
  5
  6 # shape (batch_size*head_number, sequence_length, d_k//head_number)
  7 q_reshape = K.reshape(q, (-1, q_shape[-2], q_shape[-1]))
  8 # shape (batch_size*head_number, d_k//head_number, k_sequence_length)
  9 k_reshape = K.reshape(k_transposed, (-1, k_t_shape[-2], k_t_shape[-1]))
 10
 11 # compute masked attention because some of the tokens in the sequence should be marked, e.g. <PAD>
 12 mask_attention = self.mask_attention(
 13     # core scaled dot product
 14     K.batch_dot(
 15         q_reshape,
 16         k_reshape)
 17     / sqrt_d, attn_mask)
 18
 19 # weighted summation between Q, K and V
 20 attention_heads = K.reshape(
 21     K.batch_dot(
 22         mask_attention,
 23         K.reshape(v, (-1, v_shape[-2], v_shape[-1]))),
 24     (-1, self.num_heads, q_shape[-2], q_shape[-1]))
 25
 26 # merge all of the head
 27 attention_heads_merged = K.reshape(
 28     K.permute_dimensions(attention_heads, [0, 2, 1, 3]),
 29     (-1, d_model))

```

我把源码连接贴出来，感兴趣代码的自己关注源代码哦。
代码连接：[keras_dialogue_generation_toolkit](https://github.com/tonywenuon/keras_dialogue_generation_toolkit)。

> 1. [https://www.cnblogs.com/wuliytTaotao/p/9560205.html](https://www.cnblogs.com/wuliytTaotao/p/9560205.html)

---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI3MTg3OTk5NCw1MzY5NDk0MjYsLTUyNz
YwNzQxNSwxOTYyNDMyNDIsLTEyOTQ1NTM0NTgsNDA4MDUyOTgy
LC0xODQ3ODY5MjkwLDExMzg1MDk1OSw0MDM4MzI4MzMsLTEzMT
UyMTYwNSwtMTk3NjkyMjcyMSwtMTczMDMwMzY5NiwxOTU1MDUx
MTczLDEyMTc5MjA2OTUsLTEwOTQzMDEwNzUsODgwNzI0MTUxLD
E2MzQyNjk5MTYsMTU2OTkwOTM3NCwxNzI4Njg2Njc0LDE3NDA2
MTU5NjFdfQ==
-->