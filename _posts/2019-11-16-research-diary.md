---
title: 'Research Diary'
date: 2019-11-16
permalink: /posts/2019/11/research-diary/
tags:
  - research diary
---

## Unsupervised learning / representation learning

### [ICLR 2020] Momentum Contrast for Unsupervised Visual Representation Learning [[pdf]](https://arxiv.org/abs/1911.05722)
_Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick_
- From a perspective on contrastive learning as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. This suggests that the gap between unsupervised and supervised representation learning has been largely closed in many vision tasks.
- The “keys” (tokens) in the dictionary are sampled from data (e.g., images or patches) and are represented by an encoder network. Unsupervised learning trains encoders to perform dictionary look-up: an encoded “query” should be similar to its matching key and dissimilar to others. Learning is formulated as minimizing a contrastive loss
- contrastive learning: Dimensionality reduction by learning an invariant mapping, CVPR 2006
- ![](/images/fig_research_diary/1.PNG =50x)

### [ICLR 2019] Learning deep representations by mutual information estimation and maximization [[pdf]](https://arxiv.org/abs/1808.06670)
_R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Adam Trischler, and Yoshua Bengio_

### [ICCV 2019] Local aggregation for unsupervised learning of visual embeddings [[pdf]](https://arxiv.org/abs/1903.12355)
_Chengxu Zhuang, Alex Lin Zhai, Daniel Yamins_

### [CVPR 2018 spotlight] Unsupervised feature learning via non-parametric instance discrimination [[pdf]](https://arxiv.org/pdf/1805.01978v1.pdf)
_Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin_
- ![](/images/fig_research_diary/2.PNG)

### [arXiv 2018] Representation learning with contrastive predictive coding [[pdf]](https://arxiv.org/abs/1807.03748)
_Aaron van den Oord, Yazhe Li, and Oriol Vinyals_

## Others

### [ICLR 2020] Geom-GCN: Geometric Graph Convolutional Networks [[pdf]](https://openreview.net/forum?id=S1e2agrFvS)
_Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, Bo Yang_


#### [ICLR 2018] Cem-rl: Combining evolutionary and gradient based methods for policy search
