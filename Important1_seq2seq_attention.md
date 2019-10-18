
# 【重要系列 1】对话生成基础之基于 Attention 机制的Seq2Seq Model
### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


![](https://latex.codecogs.com/gif.latex?c_i=\sum_{j}{\alpha_{ij}h_j})

#### 本文收获
* 经典 Sequence-to-Sequence 模型是怎么回事？
* Attention 机制是什么？
* Attention 机制如何和 Seq2Seq model 结合？
* 如何用 Keras 实现一个 Seq2Seq-Attention 的模型？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* <span id = "paper2">Paper 2</span>: [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。那么就进入正题吧。

说到 Sequence-to-Sequence 模型，也就是大家常说的序列到序列模型，我们需要从循环神经网络 RNN (Recurrent Neural Network) 讲起。最初的 Seq2Seq  模型是在 [Paper 1](#paper1) 中描述的。有兴趣的读者可以从这里连接到原文去看。Seq2Seq 模型已经是比较成熟的了，网上有很多人对其进行了解释，这里我按照我对它的理解，也介绍一遍（大神请指教呀）。

## 1. Sequence-to-Sequence 模型
![Sequence-to-Sequence 示例图](https://github.com/tonywenuon/posters/blob/master/images/important1/seq2seq.png?raw=true)
对于大名鼎鼎的 Encoder-Decoder 架构，Sequence-to-Sequence 就是它的一个具体实现。序列到序列模型，顾名思义输入是一个序列，输出也是一个序列。Encoder 负责把输入序列变成向量来表示整个序列的语义，Decoder 负责把这个向量解释成输出序列。我们从这张图开始吧，图里符号比较多，我分开一点点解释。

首先图中的方框框就是 RNN 了（等会介绍），先关注方框框下面的序列。图中 “ABC” 是输入序列，“WXYZ” 是输出序列（说是输出序列并不严谨，实际上是意图输出序列，也就是训练集中的内容。在实际输出中并不一定输出 “WXYZ”）。那么在训练的时候，Encoder 接收 “ABC”，Decoder 接收 “WXYZ”。再看方框框上面的字符串 “WXYZ”，这个的意思就是在每个方框框的地方，接收一个字符（从下面）再输出一个字符（到上面），例如第 5 个方框，接收 W，输出 X。注意到第 5 个方框还有一个输入箭头是从第 4 个方框来的。这就是说，根据前面的信息（第 5 个方框以前），和当前输入的信息（输入 W）来生成 X。大家有没有觉得很熟悉？没错咯，这就是语言模型。根据马尔科夫假设，当前输出只依赖于前面的输出。

那么可能有的同学会问了，**问题 1**，训练的时候，在 Decoder 的时候你是知道输入了，比如这个 W，但是预测的时候，你哪来的 ground truth 啊？那么这个问题就是语言模型的问题啦，你看第 4 个方框的输出是啥？是 W，这就是第 5 个方框的输入。在预测阶段，当预测当前字符的时候，就是拿前一个状态的输出作为当前的输入。这样在 Decoder 的时候，就可以根据一个起始输入来一个一个词的预测，直到预测结束就生成了一个序列。
**问题 2**，图里的“\<EOS\>” 是啥啊？有啥作用呢？EOS 的全称是 End of Sequence，也就是序列终止标记。在训练集中本来是不包括的，所以你在把训练集输入到模型之前是需要自己把这个标记加上的。在 Encoder 阶段，它表达的意思是告诉模型 Encoder 已经到了最后一个字符了，那么你可以把这个节点的输出作为向量输入到 Decoder 里了。所以你看在图里，模型遇到 \<EOS\> 后开始预测。而在Decoder 阶段，\<EOS\> 代表预测结束了，当预测出这个字符以后，就可以停止了，这个字符以前的所有字符就组成了输出序列。

好啦，Seq2Seq 整体上就这么多内容，是不是很简单。但是这个 RNN 本来我不想讲，但是它对于理解 Seq2Seq 又很重要，我就简述它的主要原理吧。


#### RNN
提起 RNN，普遍都会拿 LSTM 来举例子，我不详细解释 LSTM 的内部原理，这里给一个大神的解释，里面解释的清晰到位在最后还给了 GRU 的原理。把他那个帖子看完了，也就不用看这一段了。LSTM 解释：[Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。

在上面图里，一个方框框是一个 LSTM 单元，在模型的角度就是一个 timestep。但是因为 RNN 是循环神经网络，所以在实际实现上并不是顺序的，即一个 LSTM 接着一个 LSTM，而是同一个 LSTM 的不同 timestep。这里可能比较难理解，拿 Encoder 来讲，它有 4 个输入：ABC和终止标记\<EOS\>，那么 4 个输入每个都是一个 step。在 Encoder 接收 A 的时候，更新一次当前 LSTM 单元的参数，当 B 来的时候，还是同一个 LSTM 单元来处理 B。因此当最后迭代收敛的时候，是当前这一个 LSTM 的参数收敛了（并不是多少个词就有多少个 LSTM）。之所以表达成链状是为了好理解。

## 2. Attention 机制是什么？
先上 paper，原文链接在这里呦 [Paper 2](#paper2)。
Attention 是来自于计算机视觉的一个概念，引入到了 NLP 领域。本意是在看一幅图片的时候，人们的视觉是 attention 到图片的某一部分的，而不是全部。那么对于 Seq2Seq 模型呢，就是在预测的时候，本来我是根据前一个状态来预测现在的输出。有了 Attention 呢，我就还可以 pay attention 到 Encoder 里的所有词，那么也许有的词跟我关系大，有的词关系小，对于当前预测的我来讲，我就又多了输入的信息，理论上我的输出也比原来的好。
![Seq2Seq 中的 Attention 机制](https://github.com/tonywenuon/posters/blob/master/images/important1/attention.png?raw=true)
这个图里 $X_1, X_2, ..., X_T$ 就是 Encoder 的输入了，也就是上图中的 ABC。$h_1, ..., h_T$ 就是 Encoder 中每一层 LSTM 的输出，象征着每一个词的向量表示。至于这个图里的两个方向的 h，它他的意思是双向 RNN。这里就不用 care 了，忽略就好，对理解 Attention 没帮助，感兴趣的自己去搜一下就好。下面讲 $y_t$，图里当前的预测词是 $y_t$，它接收它前一个状态 $y_{t-1}$ 和图中下半部分的上下文向量 (attention)。 这里上下文向量我用 $c_i$ 来表示，就是 Decoder 里第 i 个词的 attention。重点来了，拿本本记号，要上公式了（哈哈，不用怕，不用记公式，知道公式表达啥意思就行了）。

$$c_i = \sum_{j}{\alpha_{ij}h_j}$$
$h_j$ 就是每一个 Encoder 中的第 j 个词，$\alpha_{ij}$ 是第 j 个词对应的权重。那么这个公式就很好理解了，他就是 Encoder 中每个词的向量乘以它的权重并且加到一起。这就是 Attention，即 Decoder 的时候，第 i 个词和 Encoder 中每一个词都会有个 Attention 向量。也就是 pay attention 到了 Encoder 中的某个词，是不是很好理解。那么下一个问题就是这个权重是怎么计算来的？

$$\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}{exp(e_{ik})}}$$
$$e_{ij} = F(s_{i-1}, h_j)$$

可以看到，$e_{ij}$ 就是 Decoder 中前一个状态 $s_{i-1}$ 和 Encoder 每一个词 $h_j$ 的相似度。$F(\cdot)$ 就是相似度函数，在原文 [Paper 2](#paper2) 里，这个相似度函数是个前向神经网络，当然了，你也可以把它就当成简单的 Cosine 相似度也是没有问题的。接下来呢，这个权重 $\alpha_{ij}$ 就是把各个相似度归一化啦。 

### Seq2Seq 结合 Attention
那么接下来的结合问题就很简单了，按照原文章的定义。
$$p(y_i|y_1, \dots, y_{i-1}, x) = g(y_{i-1},s_i, c_i)$$
$y_{i-1}$ 就是前一个 Decoder 的输出词，$s_i$ 是当前 LSTM 状态，$c_i$ 是生成当前词所需要的 Attention 上下文信息，$g(\cdot)$ 是变换函数。这就生成了当前的词 $y_i$。

## 3. 如何用 Keras 实现一个 Seq2Seq-Attention 的模型？
下面我们就一步步实现一个 Seq2Seq-Attention 模型吧。这里我们采用 GRU，其实原理和 LSTM都是类似的。

```python
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(hidden_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       recurrent_activation='sigmoid')
    
    def call(self, enc_inputs, hidden_state):
        x = self.embedding(enc_inputs)
        output, state = self.gru(x, initial_state=hidden_state)
        return output, state

```
总的来说，Encoder 负责接收一个输入，然后调用 GRU 来输出最后的 Encoder outputs 和 state。

```python
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(hidden_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       recurrent_activation='sigmoid')
        self.final_probs = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.hidden_dim)
        self.W2 = tf.keras.layers.Dense(self.hidden_dim)
        self.V = tf.keras.layers.Dense(1)
    def call(self, dec_input, hidden_state, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)

        # This also can be changed to dot score
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # for each input sequence and each word in the sequence, there is a corresponding weight
        attention_weights = tf.nn.softmax(score, axis=1)

        # add all encoder_output with multiplying their weights
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x is only one word of decoder. shape == (batch_size, 1, embedding_dim)
        x = self.embedding(dec_input)

        # concat attention vector and decoder input
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        probs = self.final_probs(output)

        return probs, state, attention_weights
```

代码中都有注释，我就不一句一句讲了，对照着前面介绍的 Attention 原理，大家就能知道代码是怎么回事了。本来我想试着从数据生成到训练，到预测都写一下。但是捋了一下，小弟还是觉得大家直接看代码比较好。我讲的话，如果讲的太碎，就逻辑性不强；如果讲的太泛，那跟不讲没啥区别。我把源码连接贴出来，感兴趣代码的自己看吧还是。
代码连接：[keras_dialogue_generation_toolkit](https://github.com/tonywenuon/keras_dialogue_generation_toolkit)。

---
> “知乎专栏-问答不回答”，一个期待问答能回答的专栏。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4MDY2NDU3MjAsLTMyODExNzM0NCwtMT
Q0MjUwMzkwNF19
-->
