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
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。那么就进入正题吧。

### 1. 什么是基于知识生成模型 (Knowledge-injecting model)？
大家都知道，我们人类在回答任何问题的时候都是根据我们的既有知识来回答。比如别人问你的名字，如果你不知道的话（当然啦，正常都知道啦），你没办法回答他。

![](https://github.com/tonywenuon/posters/blob/master/images/knowledge1/knowledge1.png?raw=true)


<!--stackedit_data:
eyJoaXN0b3J5IjpbNTkwNjI4NjY0LDczMjQ1OTQ0OSw4MTk3ND
g3NDVdfQ==
-->