# ReadingNotes
ACL2021 & NAACL2021 paper 阅读笔记

持续更新
[TOC]

## SLU

### GL-GIN: Fast and Accurate Non-Autoregressive Model for Joint Multiple Intent Detection and Slot Filling [[paper](https://aclanthology.org/2021.acl-long.15.pdf)][[code](https://github.com/yizhen20133868/GL-GIN)]

* problem

用非自回归模型（self attention、GAT等）加速意图识别和抽槽的推理速度

* method

提出全局局部图形交互网络（GL-GIN），其中主要的组成部分有local slot-aware graph interaction layer和global intent- slot graph interaction layer，前者学习槽之间的关系，后者学习意图和槽的交互关系
![]("fig/GL-GIN.jpg")

* conclusion & thoughts

GL-GIN与论文中的baseline比较，可以达到SOTA的效果，且推理速度快了11倍
**快速推理的联合的意图分类+抽槽模型**


Supervised Neural Clustering via Latent Structured Output Learning: Application to Question Intents [[paper](https://aclanthology.org/2021.naacl-main.263.pdf)][code]
* problem

自动识别意图

* method

聚类+结构化 优化方程

* conclusion & thoughts

oos（out of scope）的意图 召回竟然很高
结构化聚类，多领域可以放在一起做，**泛化性很高**
个人感觉，适合闲聊QA


Enhancing the generalization for Intent Classification and Out-of-Domain Detection in SLU [[paper](https://aclanthology.org/2021.acl-long.190.pdf)][code]
* problem

提出一个DRM（domain-regularized module），可以只用in-domain的数据训练 in-domain的intent分类模型和out-of-domain（ood）检测模型。

* method

DRM的loss function由两部分组成，领域分类logits（fd）和分类logits（fc），根据论文描述，推理隐含着假设IND和OOD几乎不重合，于是loss function
$$f = f_{c}/f_{d}$$

* conclusion & thoughts

一言概括，将领域拒识模型和意图分类模型统一，并且只用领域内数据训练。
这个方法可以同时增强两模型的泛化能力，但重点是检测ood/ind。
DRM可以代替最后一层linear，所以可以灵活用于神经网络模型
个人感觉，**适合只有单个领域数据时，做冷启动**，减少其他领域的干扰

Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training [[paper](https://aclanthology.org/2021.acl-long.273.pdf)] [[code](https://github.com/liam0949/DCLOOS)]


* problem

任务型对话系统中，out-of-scope的意图识别问题

* method

先生成伪out-of-scope（outliers）数据：
1. 对inliners进行变换
2. 从open domain的文本中sample

training阶段利用伪数据，训练discriminator来识别out-of-scope，提升泛化能力。testing阶段，用真实out-of-scope来训练。

* conclusion & thoughts

适用于对话系统中已有定义好的intents，如何把未知的intents（outliers）识别出来。

个人感觉，是把常规操作写成了论文，没有大的突破。



[Intent Classification and Slot Filling for Privacy Policies](https://aclanthology.org/2021.acl-long.340.pdf)

[A Semi-supervised Multi-task Learning Approach to Classify Customer Contact Intents](https://aclanthology.org/2021.ecnlp-1.7.pdf)

[Semi-supervised Meta-learning for Cross-domain Few-shot Intent Classification](https://aclanthology.org/2021.metanlp-1.8.pdf)



## ER
[Modularized Interaction Network for Named Entity Recognition](https://aclanthology.org/2021.acl-long.17.pdf)

[MECT: Multi-Metadata Embedding based Cross-Transformer for Chinese Named Entity Recognition](https://aclanthology.org/2021.acl-long.121.pdf)

[Named Entity Recognition with Small Strongly Labeled and Large Weakly Labeled Data](https://aclanthology.org/2021.acl-long.140.pdf)

[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142.pdf)

[Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition](https://aclanthology.org/2021.acl-long.216.pdf)

[A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition](https://aclanthology.org/2021.acl-long.372.pdf)

[Enhancing Entity Boundary Detection for Better Chinese Named Entity Recognition](https://aclanthology.org/2021.acl-short.4.pdf)

## knowledge distillation
[Structural Knowledge Distillation: Tractably Distilling Information for Structured Predictor](https://aclanthology.org/2021.acl-long.46.pdf)

[MATE-KD: Masked Adversarial TExt, a Companion to Knowledge Distillation](https://aclanthology.org/2021.acl-long.86.pdf)

[Marginal Utility Diminishes: Exploring the Minimum Knowledge for BERT Knowledge Distillation](https://aclanthology.org/2021.acl-long.228.pdf)

[Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains](https://aclanthology.org/2021.acl-long.236.pdf)

[Matching Distributions between Model and Data: Cross-domain Knowledge Distillation for Unsupervised Domain Adaptation](https://aclanthology.org/2021.acl-long.421.pdf)

## pre-trained LM
[BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data](https://aclanthology.org/2021.acl-long.14.pdf)

## RE
[UniRE: A Unified Label Space for Entity Relation Extraction](https://aclanthology.org/2021.acl-long.19.pdf)

## Text Style Transfer
[Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization](https://aclanthology.org/2021.acl-long.8.pdf)

## NLG
[Mention Flags (MF): Constraining Transformer-based Text Generators](https://aclanthology.org/2021.acl-long.9.pdf)
