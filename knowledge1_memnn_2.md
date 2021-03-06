# 【Knowledge-Injecting 1】根据背景知识生成答案之记忆网络 （Memory Network）

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！

> 大家都知道，我们人类在回答任何问题的时候都是根据我们的既有知识来回答。对于模型也不例外。现在已经有很多研究在基于 Knowledge 来生成回答。Knowledge-injecting 生成模型是一个系列，接下来会其他的基于背景知识的生成模型，本文是系列第一篇。

#### 本文收获
* 什么是根据背景知识生成模型？
* 什么是记忆网络 （Memory Network）？
* 如何用 Keras 实现一个 Memory Network 的模型？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [End-to-End Memory Network](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
* <span id = "paper2">Paper 2</span>: [A Knowledge-Grounded Neural Conversation Model](https://isi.edu/~ghazvini/papers/Neural_conversational_model.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。那么就进入正题吧。

### 1. 什么是基于知识的生成模型 (Knowledge-injecting model)？
大家都知道，我们人类在回答任何问题的时候都是根据我们的既有知识来回答。比如别人问你的名字，如果你不知道的话（当然啦，正常都知道啦），你没办法回答他。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/knowledge1/knowledge1.png?raw=true)<br><br>

再比如上图中的例子，用户输入“Going to Kusakabe tonight.”，背景知识里有 “Consistently the best **omakase** in San Francisco”，那么人类回答的话会回答 “Try omasake”，但是模型来回答就仅仅是 “Have a great time!” 了。

再举一个例子，来自于 Facebook 的 bAbI 数据集。

```
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary? bathroom 1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel? hallway 4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel? hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel? office 11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra? bathroom 8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra? bathroom 2
```

例子中，第一个数字代表这一行的 ID，在每一个问题之前，都是 knowledge，比如第三行，问题是 “Where is Mary？”，答案是 “bathroom”，答案的来源是 ID=1。其他的情况依次类推。是不是有点感觉了，作为模型来讲，要记住前面所有的信息才能回答相应的问题。那么到这儿，Memory Network 该出场了。

### 2. Memory Neural Network (MemNN)
对原文感兴趣的，请链接到 [Paper 1](#paper1) 去看原文。首先我还是根据原文，介绍一下 MemNN。MemNN 是用来记忆外部信息的。总的来说，**就是根据每一条外部知识和 `query` 的相关关系来把外部知识中有用的部分更新到 `query` 的向量表示中。**

#### 2.1 单层 MemNN

##### 所有输入的向量表示（背景知识+`query`）

首先对于输入的知识，要进行向量表示模型才能认识。假设 ![](https://latex.codecogs.com/gif.latex?x_1,\dots,x_i) 表示 i 个知识。那么每个知识都会被表示成向量，我们用 ![](https://latex.codecogs.com/gif.latex?m_i) 来代替这个向量。从词形到向量的过程，是一个单词映射的过程。再假设词表的大小是 V，向量维度是 d。那么我们现在就有个了 embedding 矩阵 A（维度是 d*V）通过这个矩阵 A，每个词都能映射成一个 d 维向量。知识表示完了，对于 query （理解成 bAbI 中的问题，比如 ID=3 的那行。），当然也得表示，原始模型中，用另一个 embedding 矩阵 B（维度也是 d*V）来表示。query 向量化后用 `u` 来代替。现在 `query` 和所有的背景知识都有向量表示了，那么下面的工作就是回答这个问题：如何能知道当前的 `query` 和哪个知识最相关呢？

模型角度来讲，我们通常有几种计算相关度的方法，[Paper 1](#paper1) 中使用点积相似度。

<br><br>![](https://latex.codecogs.com/gif.latex?p_i=Softmax(u^Tm_i))<br><br>

把 `u` 和每一个 ![](https://latex.codecogs.com/gif.latex?m_i) 计算相似度，最后取 Softmax，就可以得到 `query` 和知识的相关程度，用 ![](https://latex.codecogs.com/gif.latex?p_i) 来表示，![](https://latex.codecogs.com/gif.latex?p_i) 可以理解成每个知识在当前 `query` 下的权重。

##### 输出的向量表示
输入有了，那么输出什么呢？当然了，整个模型的目标输出就是 `query` 的向量表示。但是在输出的时候，怎么把知识整合到 `query` 的向量中呢？这里还需要用到知识的输出 embedding 矩阵 C（维度也是 d*V）每个知识 ![](https://latex.codecogs.com/gif.latex?x_i) 会被表示成 ![](https://latex.codecogs.com/gif.latex?c_i)。虽然 C 和 A 都是用来表示知识的向量矩阵的，但是他们不一样，一个管输入表示，一个管输出表示。

<br><br>![](https://latex.codecogs.com/gif.latex?o=\sum{p_ic_i})<br><br>

至此我们就有了一个由所有知识的加权求和的一个向量表示。那么理想情况下，重要的知识的权重就高，不相关的知识权重就低。

<br><br>![](https://latex.codecogs.com/gif.latex?u=u+o)<br><br>

由这个公式来更新 `query` 的向量，也就把外部知识的向量整合进了 `query` 中。

##### 预测结果
有了最后的 `query` 的向量表示，那么预测最后的结果就简单了，就是一个包装了 Softmax 的 MLP。
<br><br>![](https://latex.codecogs.com/gif.latex?\hat{a}=Softmax(Wu))<br><br>

这里的 W 和前面的 A，B，C 矩阵一起，都是可训练参数。在利用交叉熵损失函数计算 ![](https://latex.codecogs.com/gif.latex?\hat{a}) 和真实 label 之间的 loss 值，就可以训练出一个单层网络了。架构图请见下图 a) 部分，在图中找到对应 A，B，C 和 W 来理解架构。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/knowledge1/memnn.png?raw=true)<br><br>

#### 2.2 多层 MemNN

架构图 b) 部分呢就是多层 MemNN，为了让 MemNN 推理性更强，作者提出了多层 MemNN，从图中可以看到，其实和单层 MemNN 没区别，就是多了 ![](https://latex.codecogs.com/gif.latex?A^1)，![](https://latex.codecogs.com/gif.latex?C^1)，![](https://latex.codecogs.com/gif.latex?A^2)，![](https://latex.codecogs.com/gif.latex?C^2)，![](https://latex.codecogs.com/gif.latex?A^3)，![](https://latex.codecogs.com/gif.latex?C^3) 而已，每一层都有自己的输入输出 embedding 矩阵。

用 k 来表示第 k 层，那么在第 k 层结束的时候，都会对 u 进行更新。

<br><br>![](https://latex.codecogs.com/gif.latex?u^{k+1}=u^k+o^k)<br><br>

其他部分和单层一样。那么另一个问题就来了。假设词表大小是 50000，向量 100 维。那么每个 A，B，C 的矩阵大小每个都是 50000*100=500w 个待训练参数。一层就是 1500w，三层就是 4500w 个待训练参数，这只是 A，B，C 还没有算 W 的大小。这个代价略高，为了解决这个问题，在原文中提出了两种解决办法来减少待训练参数。

##### （1）邻接法
邻接法，顾名思义了，就是相邻的两层有某种关系，那么什么关系呢？作者规定：

* ![](https://latex.codecogs.com/gif.latex?A^{k+1}=C^k)
* ![](https://latex.codecogs.com/gif.latex?W^T=C^k)
* ![](https://latex.codecogs.com/gif.latex?B=A^1)
 
也就是相邻层之间，低层的输出和高层的输入 embedding 矩阵共享参数；最后预测层和最后层的输出 embedding 共享参数;第一层的 `query` 和外部知识共享参数。

##### （2）层复制法
这个也很好理解，就是每一层的参数都是一样的。

* ![](https://latex.codecogs.com/gif.latex?A^1=A^2=\dots=A^k)
* ![](https://latex.codecogs.com/gif.latex?C^1=C^2=\dots=C^k)

通过这两种方法，就可以极大地减少 MemNN 模型的待训练参数量了。

### 3. 如何用 Keras 实现一个 Memory Network 的模型？

首先先定义一个 Position Embedding，这个的意思是说，每个词在一个 sentence 里的顺序是不一样的，比如 “I like Jazz `music` very much. Do you like hip-hop `music`?”，这里有两个 `music` 但是第一个和 Jazz 更相关，而第二个和 hip-hop 更相关。如果不加位置区分的话，这两个 `music` 表达的向量就是相同的了，体现不了语义。

```python
class PosEncodeEmbedding(Layer):
    """
    Position Encoding described in section 4.1
    """
    def __init__(self, _type, seq_len, embedding_dim, **kwargs):
        self._type = _type
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.pe_encoding = self.__position_encoding()
        super(PosEncodeEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['_type'] = self._type
        config['seq_len'] = self.seq_len
        config['embedding_dim'] = self.embedding_dim
        config['pe_encoding'] = self.pe_encoding
        return config 

    def compute_output_shape(self, input_shape):
        return input_shape

    def __position_encoding(self):
        encoding = np.ones((self.embedding_dim, self.seq_len), dtype=np.float32)
        ls = self.seq_len + 1
        le = self.embedding_dim + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (self.embedding_dim + 1) / 2) * (j - (self.seq_len + 1) / 2)
        encoding = 1 + 4 * encoding / self.embedding_dim / self.seq_len
        # Make position encoding of time words identity to avoid modifying them 
        encoding[:, -1] = 1.0
        # shape: (seq_len, embedding_dim)
        return np.transpose(encoding)

    def call(self, inputs):
        assert len(inputs.shape) >= 2
        res = None
        if self._type == 'query':
            res = inputs * self.pe_encoding
        elif self._type == 'story':
            pe_encoding = np.expand_dims(self.pe_encoding, 0)
            res = inputs * pe_encoding
        return res
```

接下来来实现具体的 MemNN 代码。

```python


class MemNNModel:
    def __init__(self, args):
        self.args = args
        self.query_pe_embedding = PosEncodeEmbedding('query', args.src_seq_length, args.embedding_dim)
        self.story_pe_embedding = PosEncodeEmbedding('story', args.fact_seq_length, args.embedding_dim)
        self.decoder = my_gru(name='decoder_gru', units=args.hidden_dim)
        self.decoder_dense = Dense(args.vocab_size, activation='softmax', name='output_dense')
        self.decoder_embedding = Embedding(name='decoder_embedding',
                                           input_dim=self.args.vocab_size, 
                                           output_dim=self.args.embedding_dim)

    def get_config(self):
        config = super().get_config()
        config['args'] = self.args
        return config

    #we have to turn sentence into embedding not just a word
    #here we just adding embeddings up
    def __emb_sent_bow(self, inp):
        emb_op = TimeDistributed(Embedding(self.args.vocab_size, self.args.embedding_dim, name='embeddings'))
        emb = emb_op(inp) #taking each word and putting it through separate embedding, this is what Timedistributed doing 
        emb = self.story_pe_embedding(emb)  # add position encoding to story
        emb = Lambda(lambda x: K.sum(x, 2))(emb)  #then we do lambda layer to add them up
        return emb, emb_op

    def get_model(self):
        def _one_hop(emb_q, A):
            # calculate weights between query and stories
            x = Reshape((1, self.args.embedding_dim))(emb_q)
            x = Dot(axes=2)([A, x])
            x = Reshape((self.args.fact_number, ))(x)
            x = Activation('softmax')(x)
            match = Reshape((self.args.fact_number, 1))(x)

            # multiply weights to stories
            emb_story, _ = self.__emb_sent_bow(inp_story)
            x = Dot(axes=1)([match, emb_story])
            x = Reshape((self.args.embedding_dim, ))(x)
            x = Dense(self.args.embedding_dim)(x)
            # update query_embedding
            new_q = Add()([x, emb_q])
            return new_q, emb_story

        inp_story = Input(name='story_input', 
                            shape=(self.args.fact_number, self.args.fact_seq_length), 
                            dtype='int32'
                           )
        inp_q = Input(name='query_input',
                            shape=(self.args.src_seq_length, ), 
                            dtype='int32'
                           )
        
        # query and stories share the first layer parameter
        emb_story, emb_story_op = self.__emb_sent_bow(inp_story)
        emb_q = emb_story_op.layer(inp_q)
        emb_q = self.query_pe_embedding(emb_q) 
        emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)

        # from 2nd layer on, update query_embedding. Each hop, story has new embeddings
        for i in range(self.args.hops):
            if i == 0:
                # the first layer
                response, emb_story = _one_hop(emb_q, emb_story)
            else:
                response, emb_story = _one_hop(response, emb_story)

        # get the final output of the MemNN, taking as input of the following GRU Decoder
        encoder_states = response
        inp_answers = Input(name='answer_input',
                            shape=(self.args.tar_seq_length, ), 
                            dtype='int32',
                           )
        emb_answers = self.decoder_embedding(inp_answers)
        outputs, states = self.decoder(emb_answers, initial_state=encoder_states)

        # final output
        final_outputs = self.decoder_dense(outputs)

        # define model
        model = Model(inputs=[inp_q, inp_story, inp_answers], outputs=final_outputs)
        model.summary()

        # Define the inference model: Encoder
        encoder_model = Model(inputs=[inp_q, inp_story], outputs=encoder_states)

        # Define the inference model: Decoder
        inp_decoder_state = Input(shape=(self.args.embedding_dim, ))
        decoder_outputs, decoder_states = self.decoder(emb_answers, initial_state=inp_decoder_state)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(inputs=[inp_answers, inp_decoder_state], outputs=decoder_outputs)

        return model, encoder_model, decoder_model
```

上面 Keras 代码中，我们实现了一个邻接法的 MemNN，代码中的 story 相当于背景知识。
完整代码连接：[keras_dialogue_generation_toolkit](https://github.com/tonywenuon/keras_dialogue_generation_toolkit)。

---
> “知乎专栏-问答不回答”，一个期待问答能回答的专栏。

<!--stackedit_data:
eyJoaXN0b3J5IjpbNjkyMjc1MDgyLDEwODM1NjQxMzQsLTM0OD
g3Mzc2NywxNTgxNjM5MDA3LC0yNjE4NzQ2NTMsLTg4ODU1MzU4
MywtODI3NjAwMjM2LDU5MDYyODY2NCw3MzI0NTk0NDksODE5Nz
Q4NzQ1XX0=
-->