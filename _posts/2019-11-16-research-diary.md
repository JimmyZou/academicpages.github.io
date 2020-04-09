---
title: 'Research Diary'
date: 2019-11-16
permalink: /posts/2019/11/research-diary/
tags:
  - research diary
---

## Dynamic vision system (event-based vision)

### research group: [http://rpg.ifi.uzh.ch/research_dvs.html](http://rpg.ifi.uzh.ch/research_dvs.html)

### [TPAMI 2018] Event-based, 6-DOF Camera Tracking from Photometric Depth Maps [[pdf]](http://rpg.ifi.uzh.ch/docs/PAMI17_Gallego.pdf)
_Guillermo Gallego, Jon E.A. Lund, Elias Mueggler, Henri Rebecq, Tobi Delbruck, Davide Scaramuzza_

### [IJCV 2017, BMVC 2016] EMVS: Event-based Multi-View Stereo - 3D Reconstruction with an Event Camera in Real-Time [[pdf]](https://supitalp.github.io/research/publication/emvs_ijcv/)
_Henri Rebecq, Guillermo Gallego, Elias Mueggler, Davide Scaramuzza_

### [ECCV 2018] Asynchronous, Photometric Feature Tracking using Events and Frames [[pdf]](https://arxiv.org/abs/1807.09713)
_Daniel Gehrig, Henri Rebecq, Guillermo Gallego, Davide Scaramuzza_

### [CVPR 2018] A Unifying Contrast Maximization Framework for Event Cameras, with Applications to Motion, Depth and Optical Flow Estimation [[pdf]](https://arxiv.org/abs/1804.01306)
_Guillermo Gallego, Henri Rebecq, Davide Scaramuzza_

### [ECCV 2018] Semi-Dense 3D Reconstruction with a Stereo Event Camera [[pdf]](https://arxiv.org/abs/1807.07429)
_Yi Zhou, Guillermo Gallego, Henri Rebecq, Laurent Kneip, Hongdong Li, Davide Scaramuzza_

### [CVPR 2019] Events-to-Video: Bringing Modern Computer Vision to Event Cameras [[pdf]](https://arxiv.org/abs/1904.08298)
_Henri Rebecq, René Ranftl, Vladlen Koltun, Davide Scaramuzza_

### [CVPR 2019] Focus Is All You Need: Loss Functions For Event-based Vision [[pdf]](https://arxiv.org/abs/1904.07235)
_Guillermo Gallego, Mathias Gehrig, Davide Scaramuzza_

### [Scientific Report] A Spiking Neural Network Model of Depth from Defocus for Event-based Neuromorphic Vision [[pdf]](https://www.nature.com/articles/s41598-019-40064-0.pdf)
_Germain Haessig, Xavier Berthelon, Sio-Hoi Ieng & Ryad Benosman_

## Unsupervised learning / representation learning

### [ICLR 2020] Contrastive Representation Distillation [[pdf]](https://arxiv.org/abs/1910.10699)
_Yonglong Tian, Dilip Krishnan, Phillip Isola_
- Examples include distilling a large network into a smaller one, transferring knowledge from one sensory modality to a second, or ensembling a collection of models into a single estimator. **Knowledge distillation, the standard approach to these problems, minimizes the KL divergence between the probabilistic outputs of a teacher and student network.** We demonstrate that this objective ignores important structural knowledge of the teacher network. This motivates an alternative objective by which we train a student to capture significantly more information in the teacher’s representation of the data. We formulate this objective as contrastive learning. Experiments demonstrate that our resulting new objective outperforms knowledge distillation and other cutting-edge distillers on a variety of knowledge transfer tasks, including single model compression, ensemble distillation, and cross-modal transfer.
- Motivation: Representational knowledge is structured – the dimensions exhibit complex interdependencies. The original KD objective introduced in (Hinton et al., 2015, Distilling the knowledge in a neural network) treats all dimensions as independent, conditioned on the input.

### [ICLR 2020] Momentum Contrast for Unsupervised Visual Representation Learning [[pdf]](https://arxiv.org/abs/1911.05722)
_Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick_
- From a perspective on contrastive learning as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. This suggests that the gap between unsupervised and supervised representation learning has been largely closed in many vision tasks.
- The “keys” (tokens) in the dictionary are sampled from data (e.g., images or patches) and are represented by an encoder network. Unsupervised learning trains encoders to perform dictionary look-up: an encoded “query” should be similar to its matching key and dissimilar to others. Learning is formulated as minimizing a contrastive loss
- contrastive learning: Dimensionality reduction by learning an invariant mapping, CVPR 2006
- <img src="/images/fig_research_diary/1.PNG" width="500"/>

### [ICLR 2019] Learning deep representations by mutual information estimation and maximization [[pdf]](https://arxiv.org/abs/1808.06670)
_R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Adam Trischler, and Yoshua Bengio_

### [ICCV 2019] Rethinking ImageNet pre-training [[pdf]](https://arxiv.org/abs/1811.08883)
_Kaiming He, Ross Girshick, Piotr Dollár_

### [NeurIPS] Learning Representations by Maximizing Mutual Information Across Views [[pdf]](https://arxiv.org/abs/1906.00910)
_Philip Bachman, R Devon Hjelm, William Buchwalter_

### [arXiv 2019] Data-Efficient Image Recognition with Contrastive Predictive Coding [[pdf]](https://arxiv.org/abs/1905.09272)
_Olivier J. Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, Aaron van den Oord_

### [ICCV 2019] Local aggregation for unsupervised learning of visual embeddings [[pdf]](https://arxiv.org/abs/1903.12355)
_Chengxu Zhuang, Alex Lin Zhai, Daniel Yamins_

### [CVPR 2019] Self-Supervised Representation Learning by Rotation Feature Decoupling [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf)
_Zeyu Feng, Chang Xu, Dacheng Tao_
- ![](/images/fig_research_diary/3.PNG)

### [CVPR 2018 spotlight] Unsupervised feature learning via non-parametric instance discrimination [[pdf]](https://arxiv.org/pdf/1805.01978v1.pdf)
_Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin_
- ![](/images/fig_research_diary/2.PNG)
- noise constative estimation (NCE) and proximal regularization: $$J_{NCE}(\theta) = -E_{P_d}\big[\log h(i, \mathbf{v}_i^{(t-1)}) - \lambda \|\mathbf{v}_i^{(t-1)}-\mathbf{v}_i^{(t)}\|_2^2 \big] - m\cdot E_{P_n}\big[\log (1-h(i, \mathbf{v'}^{(t-1)}))\big]$$. $$P_d$$ means data distribution and $$P_n$$ means noise distribution (uniform).

### [NeurIPS 2014] Discriminative unsupervised feature learning with convolutional neural networks [[pdf]](https://arxiv.org/abs/1406.6909)
_Alexey Dosovitskiy, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, Thomas Brox_

### [arXiv 2018] Representation learning with contrastive predictive coding [[pdf]](https://arxiv.org/abs/1807.03748)
_Aaron van den Oord, Yazhe Li, and Oriol Vinyals_

## Others

### [AAAI 2020] Hybrid Graph Neural Networks for Crowd Counting [[pdf]](https://arxiv.org/abs/2002.00092)
_Ao Luo, Fan Yang, Xin Li, Dong Nie, Zhicheng Jiao, Shangchen Zhou, Hong Cheng_

### [ICLR 2020] Inductive Matrix Completion Based on Graph Neural Networks [[pdf]](https://arxiv.org/abs/1904.12058)
_Muhan Zhang, Yixin Chen_

### [NeurIPS] Quaternion Knowledge Graph Embeddings [[pdf]](https://arxiv.org/abs/1904.10281)
_Shuai Zhang, Yi Tay, Lina Yao, Qi Liu_

### [ICML 2019] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks [[pdf]](https://arxiv.org/abs/1905.11946)
_Mingxing Tan, Quoc V. Le_

### [ICLR 2020] Geom-GCN: Geometric Graph Convolutional Networks [[pdf]](https://arxiv.org/abs/2002.05287)
_Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, Bo Yang_
- Message-passing neural networks (MPNNs) have been successfully applied to representation learning on graphs in a variety of real-world applications. However, two fundamental weaknesses of MPNNs’ aggregators limit their ability to represent graph-structured data: **losing the structural information of nodes in neighborhoods and lacking the ability to capture long-range dependencies in disassortative graphs**. Few studies have noticed the weaknesses from different perspectives. From the observations on classical neural network and network geometry, we propose a novel geometric aggregation scheme for graph neural networks to overcome the two weaknesses. The behind basic idea is the aggregation on a graph can benefit from a continuous space underlying the graph. The proposed aggregation scheme is permutation-invariant and consists of three modules, node embedding, structural neighborhood, and bi-level aggregation.

#### [ICLR 2018] Cem-rl: Combining evolutionary and gradient based methods for policy search
