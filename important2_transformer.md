
# 【重要系列 1】生成模型 Transformer

### 本着互相尊重的原则，如需转载请附加原文链接，非常感谢！


#### 本文收获
* 什么是 Transformer？
* Attention 机制是什么？
* Attention 机制如何和 Seq2Seq model 结合？
* 如何用 Keras 实现一个 Seq2Seq-Attention 的模型？
#### 重要文章
* <span id = "paper1">Paper 1</span>: [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* <span id = "paper2">Paper 2</span>: [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)
---
**本文收获**和**重要文章**我先列在前面，以使得在读正文之前能有个概念。文章也会根据**本文收获**的逻辑路线来写。那么就进入正题吧。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5Njg2NjEyMjIsMTc0MDYxNTk2MV19
-->