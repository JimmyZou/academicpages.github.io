---
title: 'Research Diary'
date: 2019-11-16
permalink: /posts/2019/11/research-diary/
tags:
  - research diary
---

## Contents<a name="contents"></a>
 - [Dynamic vision system (event-based vision)](#dynamic-vision-system)
 - [Video prediction/generation/completion](#video-prediction)
 - [Unsupervised learning / representation learning](#representation-learning)
 - [Others](#others)

---

## Dynamic vision system (event-based vision)<a name="dynamic-vision-system"></a>

### research group: [http://rpg.ifi.uzh.ch/research_dvs.html](http://rpg.ifi.uzh.ch/research_dvs.html)

### resources: [https://github.com/uzh-rpg/event-based_vision_resources](https://github.com/uzh-rpg/event-based_vision_resources)

### advantages
- Event cameras have several advantages over traditional cameras: a latency in the order of microseconds, a very high dynamic range (140 dB compared to 60 dB of traditional cameras), and very low power consumption (10mW vs 1.5W of traditional cameras). Moreover, since all pixels capture light independently, such sensors do not suffer from motion blur.

### [CVPR 2020] EventSR: From Asynchronous Events to Image Reconstruction, Restoration, and Super-Resolution via End-to-End Adversarial Learning [[pdf]](https://arxiv.org/pdf/2003.07640.pdf)
_Lin Wang, Tae-Kyun Kim, and Kuk-Jin Yoon_

### [ICCV 2019] Learning an event sequence embedding for dense event-based deep stereo [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tulyakov_Learning_an_Event_Sequence_Embedding_for_Dense_Event-Based_Deep_Stereo_ICCV_2019_paper.pdf)
_Stepan Tulyakov, Francois Fleuret, Martin Kiefel, Peter Gehler, Michael Hirsch_

### [ICCV 2019] End-to-End Learning of Representations for Asynchronous Event-Based Data [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.pdf)
_Daniel Gehrig, Antonio Loquercio, Konstantinos G. Derpanis, Davide Scaramuzza_

### [CVPR 2019] Event-based High Dynamic Range Image and Very High Frame Rate Video Generation using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/abs/1811.08230)
_S. Mohammad Mostafavi I., Lin Wang, Yo-Sung Ho, Kuk-Jin Yoon_

### [CVPR 2019] Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion [[pdf]](https://arxiv.org/abs/1812.08156)
_Alex Zihao Zhu, Liangzhe Yuan, Kenneth Chaney, Kostas Daniilidis_

### [CVPR 2019] Bringing a Blurry Frame Alive at High Frame-Rate With an Event Camera [[pdf]](https://arxiv.org/abs/1811.10180)
_Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, Yuchao Dai_

### [CVPR 2019] Focus Is All You Need: Loss Functions For Event-based Vision [[pdf]](https://arxiv.org/abs/1904.07235)
_Guillermo Gallego, Mathias Gehrig, Davide Scaramuzza_

### [CVPR 2019] Event Cameras, Contrast Maximization and Reward Functions: An Analysis [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.pdf)
_Timo Stoffregen, Lindsay Kleeman_

### [TPAMI 2018] Event-based, 6-DOF Camera Tracking from Photometric Depth Maps [[pdf]](http://rpg.ifi.uzh.ch/docs/PAMI17_Gallego.pdf)
_Guillermo Gallego, Jon E.A. Lund, Elias Mueggler, Henri Rebecq, Tobi Delbruck, Davide Scaramuzza_

### [IJCV 2017, BMVC 2016] EMVS: Event-based Multi-View Stereo - 3D Reconstruction with an Event Camera in Real-Time [[pdf]](https://supitalp.github.io/research/publication/emvs_ijcv/)
_Henri Rebecq, Guillermo Gallego, Elias Mueggler, Davide Scaramuzza_

### [ECCV 2018] Asynchronous, Photometric Feature Tracking using Events and Frames [[pdf]](https://arxiv.org/abs/1807.09713)
_Daniel Gehrig, Henri Rebecq, Guillermo Gallego, Davide Scaramuzza_

### [CVPR 2018] A Unifying Contrast Maximization Framework for Event Cameras, with Applications to Motion, Depth and Optical Flow Estimation [[pdf]](https://arxiv.org/abs/1804.01306)
_Guillermo Gallego, Henri Rebecq, Davide Scaramuzza_
- We present a unifying framework to solve several computer vision problems with event cameras: motion, depth and optical flow estimation. The main idea of our framework is to find the point trajectories on the image plane that are best aligned with the event data by maximizing an objective function: the contrast of an image of warped events. Our method implicitly handles data association between the events, and therefore, does not rely on additional appearance information about the scene. In addition to accurately recovering the motion parameters of the problem, our framework produces motion-corrected edge-like images with high dynamic range that can be used for further scene analysis.

### [ECCV 2018] Semi-Dense 3D Reconstruction with a Stereo Event Camera [[pdf]](https://arxiv.org/abs/1807.07429)
_Yi Zhou, Guillermo Gallego, Henri Rebecq, Laurent Kneip, Hongdong Li, Davide Scaramuzza_

### [CVPR 2019, TPAMI 2020] Events-to-Video: Bringing Modern Computer Vision to Event Cameras [[pdf]](https://arxiv.org/abs/1904.08298) [[pdf]](https://arxiv.org/pdf/1906.07165.pdf)
_Henri Rebecq, René Ranftl, Vladlen Koltun, Davide Scaramuzza_
- Since **the output of event cameras is fundamentally different from conventional cameras**, it is commonly accepted that they require the development of specialized algorithms to accommodate the particular nature of events.
- We propose a novel recurrent network to reconstruct videos from a stream of events, and train it on a large amount of simulated event data.

### [ECCV 2018 workshop] Unsupervised event-based optical flow using motion compensation [[pdf]](https://arxiv.org/abs/1812.08156)
_Alex Zihao Zhu, Liangzhe Yuan, Kenneth Chaney, Kostas Daniilidis_

### [Scientific Report] A Spiking Neural Network Model of Depth from Defocus for Event-based Neuromorphic Vision [[pdf]](https://www.nature.com/articles/s41598-019-40064-0.pdf)
_Germain Haessig, Xavier Berthelon, Sio-Hoi Ieng & Ryad Benosman_

### [CVPR 2017] A Low Power, Fully Event-Based Gesture Recognition System [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf) [[dataset]](http://research.ibm.com/dvsgesture/)
_Arnon Amir, Brian Taba, David Berg, Timothy Melano, Jeffrey McKinstry, Carmelo Di Nolfo, Tapan Nayak, Alexander Andreopoulos, Guillaume Garreau, Marcela Mendoza, Jeff Kusnitz, Michael Debole, Steve Esser, Tobi Delbruck, Myron Flickner, and Dharmendra Modha_

### [CVPR 2019] EV-Gait: Event-Based Robust Gait Recognition Using Dynamic Vision Sensors [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_EV-Gait_Event-Based_Robust_Gait_Recognition_Using_Dynamic_Vision_Sensors_CVPR_2019_paper.pdf)
_Yanxiang Wang, Bowen Du, Yiran Shen, Kai Wu, Guangrong Zhao, Jianguo Sun, Hongkai Wen_

### [CVPR workshop 2019] DHP19: Dynamic Vision Sensor 3D Human Pose Dataset [[pdf]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Calabrese_DHP19_Dynamic_Vision_Sensor_3D_Human_Pose_Dataset_CVPRW_2019_paper.pdf)
_Enrico Calabrese, Gemma Taverni, Christopher Awai Easthope, Sophie Skriabine, Federico Corradi, Luca Longinotti, Kynan Eng, Tobi Delbruck_

[[back to top]](#contents)
---

## Video prediction/generation/completion <a name="video-prediction"></a>

### [NeurIPS 2016] Generating Videos with Scene Dynamics [[pdf]](https://arxiv.org/abs/1609.02612)
_Carl Vondrick, Hamed Pirsiavash, Antonio Torralba_
- ![](/images/fig_research_diary/4.PNG)

### [ICLR 2016] Deep multi-scale video prediction beyond mean square error [[pdf]](https://arxiv.org/abs/1511.05440)
_Michael Mathieu, Camille Couprie, Yann LeCun_

### [ICCV 2017] The Pose Knows: Video Forecasting by Generating Pose Futures [[pdf]](https://arxiv.org/abs/1705.00053)
_Jacob Walker, Kenneth Marino, Abhinav Gupta, Martial Hebert_
- ![](/images/fig_research_diary/5.PNG)

### [ECCV 2018] Deep Video Generation, Prediction and Completion of Human Action Sequences [[pdf]](https://arxiv.org/abs/1711.08682)
_Haoye Cai, Chunyan Bai, Yu-Wing Tai, Chi-Keung Tang_
- ![](/images/fig_research_diary/6.PNG)

### [NeurIPS 2018] Video Prediction via Selective Sampling [[pdf]](https://papers.nips.cc/paper/7442-video-prediction-via-selective-sampling.pdf)
_Jingwei Xu, Bingbing Ni, Xiaokang Yang_

### [ICML 2018] Stochastic Video Generation with a Learned Prior [[pdf]](https://arxiv.org/abs/1802.07687)
_Emily Denton, Rob Fergus_

### [ICLR 2018] Stochastic Variational Video Prediction [[pdf]](https://arxiv.org/abs/1710.11252)
_Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H. Campbell, Sergey Levine_

### [CVPR 2019] Learning to Film from Professional Human Motion Videos [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Learning_to_Film_From_Professional_Human_Motion_Videos_CVPR_2019_paper.pdf)
_Chong Huang, Chuan-En Lin, Zhenyu Yang, Yan Kong, Peng Chen, Xin Yang, Kwang-Ting Cheng_

### [ICCV 2019] Improved Conditional VRNNs for Video Prediction [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Castrejon_Improved_Conditional_VRNNs_for_Video_Prediction_ICCV_2019_paper.pdf)
_Lluis Castrejon, Nicolas Ballas, Aaron Courville_

### [ICCV 2019] Compositional Video Prediction [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Compositional_Video_Prediction_ICCV_2019_paper.pdf)
_Yufei Ye, Maneesh Singh, Abhinav Gupta, Shubham Tulsiani_

### [NeurIPS 2019] High Fidelity Video Prediction with Large Stochastic Recurrent Neural Networks [[pdf]](https://arxiv.org/abs/1911.01655)
_Ruben Villegas, Arkanath Pathak, Harini Kannan, Dumitru Erhan, Quoc V. Le, Honglak Lee_

### [NeurIPS 2019] Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction [[pdf]](https://papers.nips.cc/paper/8637-unsupervised-keypoint-learning-for-guiding-class-conditional-video-prediction.pdf)
_Yunji Kim, Seonghyeon Nam, In Cho, and Seon Joo Kim_

### [ICML 2020] Stochastic Latent Residual Video Prediction [[pdf]](https://arxiv.org/pdf/2002.09219)
_Jean-Yves Franceschi, Edouard Delasalles, Mickaël Chen, Sylvain Lamprier, Patrick Gallinari_

[[back to top]](#contents)
---

## Unsupervised learning / representation learning<a name="representation-learning"></a>

### [ArXiv 2020] A Simple Framework for Contrastive Learning of Visual Representations [[pdf]](https://arxiv.org/abs/2002.05709)
_Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton_

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

[[back to top]](#contents)
---

## Others<a name="others"></a>

### Deep Meta Metric Learning [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.pdf)
_Guangyi Chen, Tianren Zhang, Jiwen Lu, Jie Zhou_

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

#### [NeurIPS 2019] General E(2) - Equivariant Steerable CNNs [[pdf]](https://arxiv.org/abs/1911.08251)
_Maurice Weiler, Gabriele Cesa_

#### [CVPR 2020] DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes [[pdf]](https://arxiv.org/abs/2004.01002)
_Jonas Schult, Francis Engelmann, Theodora Kontogianni, Bastian Leibe_

#### [SIGGRAPH 2020] CNNs on Surfaces using Rotation-Equivariant Features [[pdf]](https://arxiv.org/abs/2006.01570)
_Ruben Wiersma, Elmar Eisemann, Klaus Hildebrandt_


[[back to top]](#contents)
---
