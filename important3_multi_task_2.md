# 【重要系列 3】之多任务学习（Multi-Task）

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是多任务学习（Multi-Task）？
* 有几种多任务学习？
* 什么样的 Task 适合多任务学习？
* 如何用多任务学习实现一个端到端模型（End-to-End）？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [MULTI-TASK SEQUENCE TO SEQUENCE LEARNING](https://arxiv.org/pdf/1511.06114.pdf)
* <span id = "paper2">Paper 2</span>: [A Knowledge-Grounded Neural Conversation Model](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。

### 1. 什么是多任务学习？

多任务学习（Multi-task learning, MTL）用文字解释起来比较抽象，这里先用文字进行阐述一下，后面用例子和图片来解释就比较直观了。他是一个非常重要的机器学习模式，他旨在用其他相关任务来提升主要任务的泛化能力。简单说来多任务学习是一种集成学习方法（ensemble approach），通过对几个任务同时训练而使得多个任务互相影响。当然了，这种影响是很隐晦的影响，一般是影响在共享参数上。多个任务共享一个结构，这个结构里面的参数在优化的时候会被所有任务影响。这样在所有任务收敛的时候，这个结构就相当于融合了所有任务，因此他的泛化能力一般而言是比单任务要好的。[Paper 1](#paper1) 里详细地介绍了几种把多任务学习和 Seq2Seq 学习相结合的方法。接下来主要介绍这部分吧。

### 2. 序列到序列学习（Sequence-to-Sequence Learning）

这篇文章简单回顾一下什么是 Seq2Seq，关于 Seq2Seq 的详细解释，请跳到 [# 【重要系列 1】对话生成基础之基于 Attention 机制的Seq2Seq Model](https://zhuanlan.zhihu.com/p/87961308) 。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/important3/multi_task1.png?raw=true)<br><br>

如图所示，这是两个 Seq2Seq 的任务，左边的是机器翻译任务，右边的是句法解析任务。这里的基本思想是，Encoder 端输入 “I am a student”，Decoder 端输出另外一个序列（翻译成另一个语言，或者句法解析）。那么 Seq2Seq 怎么和多任务学习相结合呢？我们继续往下看。

### 3. 一对多 （one to many）
在 Seq2Seq 架构里，一对多的意思是一个 encoder 和多个 decoder。拿下图来举例子。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/important3/one2many.png?raw=true)<br><br>

这就是一个典型的一对多的例子。这里包含了三个任务：

* 英语-德语翻译任务
* 句子结构解析任务
* 英语-英语的 Autoencoder 任务

其中 Autoencoder 的意思是用英语自己生成自己。图中，`English` 部分是 Encoder，三个任务共享一个 `English` Encoder，因为他们的输入都是一样的。

### 4. 多对一（many to one）

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/important3/many2one.png?raw=true)<br><br>

这里面也包含三个任务：

* 德语-英语翻译任务
* 图片标题生成任务
* 英语-英语 Autoencoder 任务

这里和上面的区别是，Decoder 的参数是共享的。

### 5. 多对多（many to many）

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/important3/many2many.png?raw=true)<br><br>

多 Encoder，多 Decoder。这种任务也是比较常见的。比如图中的例子，本来是个“德语-英语”翻译的单任务。在没有其他任何额外语料的情况下，如何能提高效果呢？图中的这种集成模型，一般都会比单模型好。图中除了“德语-英语”的翻译任务以外，还有“英语-英语”，“德语-德语”的 Autoencoder 任务。Autoencoder 能够帮助一种语言更好的在语义层面表达自己。所以想象一下，之前的单任务是只处理两个语言之间的关系。多任务则是在自己更好的表达自己的情况下，处理好和另一种语言之间的关系，那么显然后者的效果会比前者好。

### 6. 什么样的 Task 适合多任务学习？
从上面的介绍，大家应该也有中感觉了，就是多任务学习其实是某种程度上迁移学习，即利用其它信息，或者其他来源的信息来影响主任务，使得当前的主任务表现更好。而方式则是通过影响共享模块的参数。那么问题是，什么样的人物适合多任务学习呢？这里就完全是个人经验了，不会完全正确，欢迎交流指正。概括说来，在 Encoder-Decoder 架构下，凡是 Encoder 或者 Decoder 有共享潜质的任务都可以用多任务学习。例如前面的一对多的例子，输入都是英文，而英文我们只要维护一个 vocabulary，那么在 Encoder 端就具有共享潜质了。同时，可以总结出的一个规律是，加入 Autoencoder 任务，一般都会有正向的效果。这是因为 Autoencoder 帮助表达自己更好，在这样的前提下，如果模型还能收敛的很好，那么结果一般而言都不错。

### 7. 如何用多任务学习实现一个端到端模型（End-to-End）？

我们引用 [paper 2](#paper2) 来实现一个基于多任务学习的端到端模型。Paper 2 主要目的是在基本的端到端模型，即`Question-Answer` pair 中加入背景知识。那么端到端模型，只有一个输入 Encoder，一个输出 Decoder，如何直接加入背景知识呢？作者利用了多任务学习模型。在文章里建立了三个任务：

* **Facts task**。利用 question 作为 query，先从知识库中检索出一个 fact 集合。把 question 和 fact 都通过 Encoder 计算后的结果，利用 Memory Network 整合到一起作为 Encoder 的输出 （Memory Network 请参见 [# 【Knowledge 生成模型 1】根据背景知识生成答案之记忆网络（Memory Network）](https://zhuanlan.zhihu.com/p/88217530)）。而 Decoder 的部分就是 ground truth 的 response 了。
* **NoFacts task**。这里就是正常的 Seq2Seq model。输入是 question，输出是 ground truth 的 response。
* **Autoencoder**。整合 question 和 fact 的 Encoder 输出作为最终的 Encoder 输出，再以 fact 作为 Decoder 的输出来训练。

这三种的架构如图所示，这个多任务学习，共享的是 Decoder 的参数。

<br><br>![](https://github.com/tonywenuon/posters/blob/master/images/important3/knowledge_ms.png?raw=true)<br><br>

主要代码可以见下面代码块。

```python
  1 # facts input corresponding to the question
  2 inp_fact = Input(name='fact_input',
  3                     shape=(self.args.fact_number, self.args.src_seq_length),
  4                     dtype='int32'
  5                    )
  6 # question input
  7 inp_q = Input(name='query_input',
  8                     shape=(self.args.src_seq_length, ),
  9                     dtype='int32'
 10                    )
 11
 12 # ground truth response input
 13 inp_tar = Input(name='tar_input',
 14                     shape=(self.args.tar_seq_length, ),
 15                     dtype='int32',
 16                    )
 17 # task 3, the target of autoencoder
 18 inp_fact_tar = Input(name='fact_tar_input',
 19                     shape=(self.args.tar_seq_length, ),
 20                     dtype='int32',
 21                    )
 22
 23 enc_output1, enc_state1 = self.s2s_encoder(inp_q)
 24 enc_state2 = self.memnn_encoder1([inp_q, inp_fact])
 25 enc_state3 = self.memnn_encoder2([inp_q, inp_fact])
 26
 27 emb_ans = self.decoder_embedding(inp_tar)
 28 emb_fact_ans = self.decoder_embedding(inp_fact_tar)
 29
 30 # task 1: seq2seq, input: question; output: answer
 31 output1, state1 = self.decoder(emb_ans, initial_state=enc_state1)
 32 # task 2: memnn, input: question and facts; output: fact
 33 output2, state2 = self.decoder(emb_fact_ans, initial_state=enc_state2)
 34 # task 3: memnn, input: question and facts; output: answer
 35 output3, state3 = self.decoder(emb_ans, initial_state=enc_state3)
 36
 37 # final output
 38 final_output1 = self.decoder_dense1(output1)
 39 final_output2 = self.decoder_dense2(output2)
 40 final_output3 = self.decoder_dense3(output3)
 41
 42 # define model
 43 model = Model(
 44     inputs=[inp_q, inp_tar, inp_fact_tar, inp_fact],
 45     outputs=[final_output1, final_output2, final_output3]
 46 )

```
其中 `self.s2s_encoder` 是 Seq2Seq 的 Encoder 模块；`self.memnn_encoder1` 是 Memory Network 的 Encoder 模块。

完整代码请参考：[keras_dialogue_generation_toolkit](https://github.com/tonywenuon/keras_dialogue_generation_toolkit)。

---
> [“知乎专栏-问答不回答”](https://zhuanlan.zhihu.com/question-no-answer)，一个期待问答能回答的专栏。




<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwNTUwMjU2NiwxNjA4MDQ5NjM5LDgwMj
M0ODg1LC0xNjcxNDcwNTMzXX0=
-->