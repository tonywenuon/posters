# 【Knowledge-Injecting 1】根据背景知识生成答案之记忆网络 （Memory Network）

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是根据背景知识生成模型？
* 什么是记忆网络 （Memory Network）？
* 如何用 Keras 实现一个 Memory Network 的模型？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [End-to-End Memory Network](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
* <span id = "paper2">Paper 2</span>: [A Knowledge-Grounded Neural Conversation Model](https://isi.edu/~ghazvini/papers/Neural_conversational_model.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。接下来会写一个 Knowledge-Injecting 系列，本文是第一篇。那么就进入正题吧。

### 1. 什么是基于知识生成模型 (Knowledge-injecting model)？
大家都知道，我们人类在回答任何问题的时候都是根据我们的既有知识来回答。比如别人问你的名字，如果你不知道的话（当然啦，正常都知道啦），你没办法回答他。

![](https://github.com/tonywenuon/posters/blob/master/images/knowledge1/knowledge1.png?raw=true)

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
对原文感兴趣的，请链接到 [Paper 1](#paper1) 去看原文。首先我还是根据原文，介绍一下 MemNN。MemNN 是用来记忆外部信息的。

#### 2.1 单层 MemNN 知识的向量表示

首先对于输入的知识，要进行向量表示模型才能认识。假设 $x_1,\dots,x_i$ 表示 i 个知识。那么每个知识都会被表示成向量，我们用 $m_i$ 来代替这个向量。从词形到向量的过程，是一个单词映射的过程。再假设词表的大小是 V，向量维度是 d。那么我们现在就有个了 embedding 矩阵 A（维度是 d*V）通过这个矩阵 A，每个词都能映射成一个 d 维向量。知识表示完了，对于 query （理解成 bAbI 中的问题，比如 ID=3 的那行。），当然也得表示，原始模型中，用另一个 embedding 矩阵 B（维度也是 d*V）来表示。query 向量化后用 `u` 来代替。现在 `query` 和所有的背景知识都有向量表示了，那么下面的工作就是回答这个问题：如何能知道当前的 `query` 和哪个知识最相关呢？

模型角度来讲，我们通常有几种计算相关度的方法，[Paper 1](#paper1) 中使用点积相似度。

$$p_i=Softmax(u^Tm_i)$$

把 `u` 和每一个 $m_i$ 计算相似度，最后取 Softmax，就可以得到
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyMzYyMDE3NzQsLTg4ODU1MzU4MywtOD
I3NjAwMjM2LDU5MDYyODY2NCw3MzI0NTk0NDksODE5NzQ4NzQ1
XX0=
-->