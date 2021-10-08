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

---

### Supervised Neural Clustering via Latent Structured Output Learning: Application to Question Intents [[paper](https://aclanthology.org/2021.naacl-main.263.pdf)][code]
* problem

自动识别意图

* method

聚类+结构化 优化方程

* conclusion & thoughts

oos（out of scope）的意图 召回竟然很高
结构化聚类，多领域可以放在一起做，**泛化性很高**
个人感觉，适合闲聊QA

---

### Enhancing the generalization for Intent Classification and Out-of-Domain Detection in SLU [[paper](https://aclanthology.org/2021.acl-long.190.pdf)][code]
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

---

### Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training [[paper](https://aclanthology.org/2021.acl-long.273.pdf)] [[code](https://github.com/liam0949/DCLOOS)]


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

---

### Intent Classification and Slot Filling for Privacy Policies [[paper](https://aclanthology.org/2021.acl-long.340.pdf)][[code](https://github.com/wasiahmad/PolicyIE)]

* problem

司法领域的意图分类和抽槽联合模型

* method

构造了一个司法领域的数据集（英文），基于这个数据集，用两种模型，验证意图分类和抽槽的效果：

1. 序列标注的意图和槽联合模型。有两种标注方式。
2. Seq2Seq的意图和槽联合模型。

* conclusion & thoughts

实验效果一般。主要的工作是数据集的构造，提供了一个benchmark，以及对实验结果的分析。模型感觉就是信息抽取。

---

### A Semi-supervised Multi-task Learning Approach to Classify Customer Contact Intents [[paper](https://aclanthology.org/2021.ecnlp-1.7.pdf)][code]

* problem

（亚马逊）电商场景下，真实数据集合由正例、多标签负例（没有明确意图）、无标签数据组成，这就会带来一系列问题。本文不用多标签分类，提出一种般监督多任务的模型，与多标签分类的baseline相比，可以提升20%

* method

1. 负例和正例的多任务训练
其他意图分类的数据只是正例（有明确意图），不包括负例（无明确意图），这就导致缺少这部分信息。
本文的做法是保留负例，用负例和正例对每个意图做二分类，作为子任务。加上全意图多标签分类作为另一个子任务
即 多任务=n个意图二分类+1个全意图多分类

2. 利用现有数据，训练了一个领域的albert

3. 利用无标签数据（这部分没仔细看）

* conclusion & thoughts

他这种负例比较有意思，用户说了一句话，给出回复，再向用户确认是否这这个意图，如果选“no”，就归为负例。

因此，这种负例还不是简单的“不属于明确意图”，而是“明确不属于某一意图”。

实验结果，auc roc可以提高20%。

---

## recommend

### CRSLab: An Open-Source Toolkit for Building Conversational Recommender System (alibaba) [[paper](https://aclanthology.org/2021.acl-demo.22.pdf)][[code](https://github.com/RUCAIBox/CRSLab)]

* motivation
缺乏统一规整的对话推荐系统的一套流程或者方法，而且由于场景数据的复杂性，很难快速做出个baseline以供后面优化对比。所以做了这个工具，可扩展性高。

* realted work
相比推荐系统，对话推荐系统是在多轮的对话中挖掘用户偏好，再结合历史信息做出推荐。
（商品）推荐和对话是对话推荐系统的主要的两个子任务。

* CRSLab
这个工具目前包含6个人工标注的数据集，和19个模型。目测基本的东西像数据处理、模型、评价都已经封装好了。
这个工具可以快速搭一个模型也可以利用这个工具定义自己的模型，只需要继承封装的类。

---

### IFlyEA: A Chinese Essay Assessment System with Automated Rating, Review Generation, and Recommendation [[paper](https://aclanthology.org/2021.acl-demo.29.pdf)]

* motivation

填补中文母语写作者的自动作文评审这个领域空白

* method
三大模块：
1. 数据处理（数据处理、各种数据集）
2. 分析模块（语法、修辞、论述）

语法层面的分析包括拼写纠错、语法纠错。

拼写纠错构造混淆集，首先，用5ngram的语言模型对替换成混淆集的词的句子打分，如果降低则纳入候选；然后用bert重新排序（这里没有说清楚怎么做的）。

语法纠错是用序列标注来预测那些位子存在错误，ResELCECTRA编码句子。

修辞层面的分析包括优美评价（训练一个二分分类器，ps有数据就是好）、比喻识别（多任务【比喻句子识别和比喻关系抽取】的分类器）、拟人识别（较困难）、排比识别（随机森林）、引用识别（信息检索、语义匹配）。

论述层面的分析包括议论文结构分析、叙事文（等级多任务分类器）、异常检测（用检索做剽窃检测）与内容分析（离题分析（困难且没说怎么做）、体裁分类、主题分类）。

3. 应用模块（打分、评语、推荐）

打分融合各种特征做优良中差的分类器。

评语生成，是根据各个维度的打分情况，对筛选的模版进行加工，输出对优缺点均有评价的评语。

推荐，构建了一个高分文章库，根据分析模块的结果来进行推荐（也没说清楚）。

* conclusion & thoughts

总体来说，评价维度非常综合全面了。可以感觉到做了很多实验，确实有数据辅助确实厉害。不足是对真正的难点都没有进行介绍，这也可以理解，毕竟要保密。

面向中小学生，也就是说理解力不强，那么这个阶段的作文水平应该也就比中文学习者的作文水平高一丢丢


---

## QA

### UnitedQA: A Hybrid Approach for Open Domain Question Answering [[paper](https://aclanthology.org/2021.acl-long.240.pdf)]

* motivation

现有open-QA的研究方法要么专注提升检索要么专注提升生成，本文提出一种能同时提升这两个方面的方法。

* method

主要包括三个模块：
1）retrieval根据qury检索出相关的篇章。检索方法：BM25【篇章处理成BOW，按照词的出现频率进行检索】、DPR（dense passage retrieval）【用bert生成passage和qury的vectors，两者乘积作为相关性得分】
2）hybrid readers根据相关篇章生成候选回答。generative reader是一个seq2seq的模型（T5）。extractive reader是一个transformer-based模型（electra）用于得到answer spans的概率/可能性/得分。并且对这两个reader都做了提升，细节见论文。
3）re-ranking模块通过线性插值合并候选并给出最终答案

* thoughts

做的优化和涉及的方法确实是很多。相比baseline提升明显，不过计算效率是之前的3倍，有待提升。


---
## Text Classification

### Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classiﬁcation [[paper](https://aclanthology.org/2021.acl-long.337.pdf)]

* 本文提出了融合输入文本和标签文本语义相似度的层次化分类模型HiMatch, 在RCV1-V2,WOS和EURLEX-57K上达到SOTA表现

---

## sentiment/emotion detection

### DCR-Net: A Deep Co-Interactive Relation Network for Joint Dialog Act Recognition and Sentiment Classification [[paper](https://arxiv.org/pdf/2008.06914.pdf)]

* motivation

现有很多对话行为和情感识别的做法，是分开做或者联合但是没有深层交互仅共享参数。本文提出的方法DCR-Net增加了两者的交互。

* method

通过co-interactive layer来做两者的交互。这一过程：1）拼接两者的表示，2）经过MLP，3）co-attention获取两者共同的主要信息，4）分别给两个decoder解码

* thoughts

这方法思路很明白，其中的组件也都是别人提出的，效果上是有提升。
感觉可以用在意图识别+身份识别的交互上。

### Directed Acyclic Graph Network for Conversational Emotion Recognition [[paper](https://aclanthology.org/2021.acl-long.123/)][[code](https://github.com/shenwzh3/DAG-ERC)]

* motivation

把有向无环图引入情感识别的任务，能够更好的对对话建模。仅仅graph-based的方法或rnn-based的方法都有其缺点，前者缺少远处信息和序列信息，后者倾向于紧邻的信息，因此，将有向无环图与rnn-based深度学习网络相结合，也可以提升对对话建模的效果。

相比GCN、GAT，DAG


* thoughts

这个方法适合多人对话的情况，像客服助理这种1v1的，DAG恐怕没有太大的提升。


Topic-Driven and Knowledge-Aware Transformer for Dialogue Emotion Detection [[paper](https://aclanthology.org/2021.acl-long.125/)]

Towards Emotional Support Dialog Systems [[paper](https://aclanthology.org/2021.acl-long.269/)]这篇的领域有点偏冷门了

### DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations [[paper](https://aclanthology.org/2021.acl-long.547/)][[code](https://github.com/zerohd4869/DialogueCRN)]

 * motivation

现有的做法没有提取、整合情感线索（emotion clue）的。
本文在认知理论的启发下，提出一个多轮迭代的提取、整合情感线索（句子层面+说话人层面）的推理模块。

 * method

主要分为三个部分：
1）分别用LSTM提取sentence-level和speaker-level的表示；
2）用多轮推理模块对上面的信息进一步推理。提取其实就是通过做attention来匹配检索，整合就是用LSTM多轮迭代；
3）级联，分类

 * thoughts

 从实验就过上看，相比其他方法提升明显。仔细一想，常用的手段都变成模拟xx理论，确实是说的很有道理的样子，厉害。

---

## NER

[Modularized Interaction Network for Named Entity Recognition](https://aclanthology.org/2021.acl-long.17.pdf)

[MECT: Multi-Metadata Embedding based Cross-Transformer for Chinese Named Entity Recognition](https://aclanthology.org/2021.acl-long.121.pdf)

### Named Entity Recognition with Small Strongly Labeled and Large Weakly Labeled Data [[paper](https://aclanthology.org/2021.acl-long.140.pdf)]

* motivation

之前的弱监督还是主要依赖有标签数据，本文选择了一个更实际的背景，即少量有标签数据+大量弱标签（从无标签中生成）。过程中发现，弱标签由于噪声过大，几乎对模型没有提升，因此提出了多阶段框架NEEDLE。

* method

NEEDLE包含三个关键点：1）弱标签补全，2）noise-aware loss，3）最后在有标签数据上微调。
主要流程有三个：1）用大量无标签数据pretrain语言模型，2）用knowledge base生成弱标签，然后在“弱标签补全”和“noise-aware loss”下用弱标签和强标签继续pretrain，3）在强标签上微调。

1. 弱标签补全： 当弱标签不是“O”的时候不变，如果是”O“，则用第一步的语言模型+随即初始化的crf的结果来替换这个”O“
2. noise-aware loss：



[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142.pdf)

[Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition](https://aclanthology.org/2021.acl-long.216.pdf)

[A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition](https://aclanthology.org/2021.acl-long.372.pdf)


### Enhancing Entity Boundary Detection for Better Chinese Named Entity Recognition [[paper](https://aclanthology.org/2021.acl-short.4.pdf)] [[code](https://github.com/cchen-reese/Boundary-Enhanced-NER)]

* motivation

没啥特别的，就是提高中文ner效果

* method

从以下两个方面，进行增强：
1. 增加一层GAT来增强短语之间的联系
2. 增加头-尾预测的子任务

模型结构，decoder是crf，encoder有三个部分：1）GRU，2）Star-transforer，3）GAT

* conclusion

从实验结果上看，效果还可以。

___

## knowledge distillation

### Structural Knowledge Distillation: Tractably Distilling Information for Structured Predictor[[paper](https://aclanthology.org/2021.acl-long.46.pdf)][[code](https://github.com/Alibaba-NLP/StructuralKD)]

* problem

对于结构化预测问题，输出空间是指数大小的，因此交叉熵损失很难直接计算和优化

* method

对结构化蒸馏的损失函数进行因式分解

* conclusion & thoughts

---

### MATE-KD: Masked Adversarial TExt, a Companion to Knowledge Distillation[[paper](https://aclanthology.org/2021.acl-long.86.pdf)]

* motivation

就是能提升student效果的蒸馏方法

* method

只用输出logits作为teacher，再加上对抗样本进行对抗训练，没什么特别的

* conclusion

感觉是工作时用的很多人都会用的方法，个人感觉这篇论文没有实质创新

---

### Marginal Utility Diminishes: Exploring the Minimum Knowledge for BERT Knowledge Distillation [[paper](https://aclanthology.org/2021.acl-long.228.pdf)][[code](https://github.com/llyx97/Marginal-Utility-Diminishe)]

* problem

HSK在蒸馏中非常重要，可以显著提升student的效果，但是随着蒸馏的HSK变多，提升的效果会下降（边际效用递减）。

* method

本文对这一现象进行研究，对HSK进行三个维度的分解（depth、width、length）

在此基础上还提出了一套蒸馏方法。

* conclusion

分析得到的结论是：1）提取蒸馏关键的HSK，即可提高student的效果；2）使用少量的HSK即可得到使用很多HSK一样的效果。

提出的蒸馏方法，可以提升蒸馏的训练速度2～3倍



---

### Matching Distributions between Model and Data: Cross-domain Knowledge Distillation for Unsupervised Domain Adaptation[[paper](https://aclanthology.org/2021.acl-long.421.pdf)][code]

 * problem

UDA无监督领域自适应常用的做法是，利用数据和共享跨域网络的结构，来调整目标网络。这种做法的弊端在于，存在源数据风险，并且灵活性很低。

 * method

提出一种不需要任何源数据的方法，CdKD，只需要一个训练好的源模型即可。并且目标模型可以采用很多结构。

UDA领域中，首个引入梯度信息。

除了蒸馏的方法和步骤以外，用**JKSD**来match两分布。

 * conclusion

 亮点，跨领域迁移/蒸馏，不需要源数据。且从实验结果上看，效果很好，明显优于其他方法。

 ---

## dialogue generation

### BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data [[paper](https://aclanthology.org/2021.acl-long.14.pdf)]

* problem

个性化对话生成缺乏大量语料，来训练一个鲁棒性和一致性强的模型

* method

提出BoB模型（BERT-overt-BERT）来解决这个问题，将个性化对话生成划分成两个子任务，用到的数据是有限的个性对话数据和。
模型包含1个bert encoder和2个bert decoder，其中一个decoder用于生成回复，另一个decoder用于一致性理解

---

## RE

[UniRE: A Unified Label Space for Entity Relation Extraction](https://aclanthology.org/2021.acl-long.19.pdf)

## Text Style Transfer

[Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization](https://aclanthology.org/2021.acl-long.8.pdf)

## NLG
[Mention Flags (MF): Constraining Transformer-based Text Generators](https://aclanthology.org/2021.acl-long.9.pdf)

## MRC

### Adversarial Training for Machine Reading Comprehension with Virtual Embeddings (*SEM 2021)

* realted work
 * 对抗训练可以提升泛化能力
 * 对抗样本
   * 图像：干净样本+扰动（梯度上升的噪声）
   * nlp：embedding+扰动（梯度上升的噪声）
   * 区别：nlp是加在embedding上，而cv是加在原始数据上

* motivation
 标准做法，对于同一个词的增加的扰动是一样的，会一定程度上忽略不同词义的区别

* method

两个embedding 提取篇章和query，都是virtual的，只用来收集扰动

## CSC

### Dynamic Connected Networks for Chinese Spelling Check (Findings of ACL 2021)

* related work

1. FASPell（爱奇艺），一个bert做检出和纠正，引入了字音和字形（相似度等得分，直接加入）
2. SpellGCN（阿里），字音字形通过GCN邻接矩阵做成一个权重矩阵，来预测最终的字
3. Soft-masked BERT（字节),没有加入字音字形，把检出和纠正做联合训练
4. PHMOSSpell（腾讯），二次pretrain引入多模态信息

* motivation

bert非自回归，依赖输出独立性假设，因此会导致输出的字之间不流畅

* method

对候选的相邻字的依赖关系建模：
把字a、字b、以及对应两个embedding，4个东西通过transformer打分

预测时用viterbi选择最优路径

* contribution

相比传统所有词的打分，生成候选的数量变少了，并且打分考虑到相邻字。
结果上达到sota。
