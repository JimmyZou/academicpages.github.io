---
title: 'Paper list of human pose and shape estimation'
date: 2019-11-12
permalink: /posts/2019/11/human-pose-shape-estimation/
tags:
  - paper list
  - human pose and shape
---

# Related page
#### https://github.com/wangzheallen/awesome-human-pose-estimation

# Paper list of human pose and shape estimation
---
## 2D human pose estimation
---
#### [ICCV 2019] Single-Stage Multi-Person Pose Machines [[pdf]](https://arxiv.org/abs/1908.09220)
_Xuecheng Nie, Jianfeng Zhang, Shuicheng Yan, Jiashi Feng_




## 3D human pose estimation
---
#### [arxiv 2019] Convex Optimisation for Inverse Kinematics [[pdf]](https://arxiv.org/abs/1910.11016)
_Tarun Yenamandra, Florian Bernard, Jiayi Wang, Franziska Mueller, Christian Theobalt_


#### [arxiv] Geometric Pose Affordance: 3D Human Pose with Scene Constraints [[pdf]](https://arxiv.org/abs/1905.07718)
_Zhe Wang, Liyan Chen, Shaurya Rathore, Daeyun Shin, Charless Fowlkes_

#### [BMVC 2019] MocapNET: Ensemble of SNN Encoders for 3D Human Pose Estimation in RGB Images [[pdf]](http://users.ics.forth.gr/~argyros/mypapers/2019_09_BMVC_mocapnet.pdf) [[CODE]](https://github.com/FORTH-ModelBasedTracker/MocapNET)
_Qammaz, Ammar and Argyros, Antonis A_

#### [3DV 2019] Multi-Person 3D Human Pose Estimation from Monocular Images [[pdf]](https://arxiv.org/abs/1909.10854)   
_Rishabh Dabral, Nitesh B Gundavarapu, Rahul Mitra, Abhishek Sharma, Ganesh Ramakrishnan, Arjun Jain_

#### [arxiv 2019] C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion [[pdf]](https://arxiv.org/abs/1909.02533)  
_David Novotny, Nikhila Ravi, Benjamin Graham, Natalia Neverova, Andrea Vedaldi_

#### [ICCV 2019] A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image [[pdf]](https://arxiv.org/abs/1908.09999)[[CODE]](https://github.com/zhangboshen/A2J)   
_Fu Xiong, Boshen Zhang, Yang Xiao, Zhiguo Cao, Taidong Yu, Joey Tianyi Zhou, Junsong Yuan_

#### [ICCV 2019] Optimizing Network Structure for 3D Human Pose Estimation [[pdf]](https://chunyuwang.netlify.com/)  
_Hai Ci, Chunyu Wang, Xiaoxuan Ma, Yizhou Wang_

#### [ICCV 2019] Cross View Fusion for 3D Human Pose Estimation [[pdf]](https://chunyuwang.netlify.com/img/ICCV_Cross_view_camera_ready.pdf) [[CODE]](https://github.com/microsoft/multiview-human-pose-estimation-pytorch)
_Haibo Qiu, Chunyu Wang, Jingdong Wang, Naiyan Wang, Wenjun Zeng_


#### [CVPR 2019] Exploiting temporal context for 3d human pose estimation in the wild [[pdf]](https://arxiv.org/abs/1905.04266)
_Anurag Arnab, Carl Doersch, Andrew Zisserman_
- bundle-adjustment-based algorithm for recovering accurate 3D human pose andmeshes from monocular videos

#### [CVPR 2018] 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning [[pdf]](https://arxiv.org/abs/1802.09232)
_Diogo C. Luvizon, David Picard, Hedi Tabia_
- a multitask framework for jointly 2D and 3D pose estimation from still images and human action recognition from video sequences

#### [CVPR 2018] Ordinal Depth Supervision for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1805.04095)
_Georgios Pavlakos, Xiaowei Zhou, Kostas Daniilidis_
- uses ordinal depth relations of human joints for 3D human pose estimation to bypass the need for accurate 3D ground truth

#### [ECCV 2018] Exploiting temporal information for 3D pose estimation [[pdf]](https://arxiv.org/abs/1711.08585)
_Mir Rayat Imtiaz Hossain, James J. Little_
- utilizes the temporal information across a sequence of 2D joint locations to estimate a sequence of 3D poses

#### [ECCV 2018] Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1804.01110)
_Helge Rhodin, Mathieu Salzmann, Pascal Fua_
- learns a geometry-aware representation using unlabeled multi-view images and then maps to 3D human pose

#### [AAAI 2018] Learning Pose Grammar to Encode Human Body Configuration for 3D Pose Estimation [[pdf]](https://arxiv.org/abs/1710.06513)
_Haoshu Fang, Yuanlu Xu, Wenguan Wang, Xiaobai Liu, Song-Chun Zhu_
- proposes a pose grammar to explicitly incorporate a set of knowledge regarding human body configuration as a generalized mapping function from 2D to 3D


#### [CVPR 2017] Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose [[pdf]](https://arxiv.org/abs/1611.07828)
_Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis_
- proposes a fine discretization of the 3D space around the subject and a coarse-to-fine prediction scheme to predict per voxel likelihoods for each joint


#### [CVPR 2017] A simple yet effective baseline for 3d human pose estimation [[pdf]](https://arxiv.org/abs/1705.03098)
_Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little_
#### [TPAMI 2017] A simple, fastand highly-accurate algorithm to recover 3d shape from 2d landmarks on a single image [[pdf]](https://arxiv.org/abs/1609.09058)
_Ruiqi Zhao, Yan Wang, Aleix Martinez_
- “lifting” ground truth 2d joint locations to 3d space is a task that can be solved with a remarkably low error rate with a relatively simple deep feed-forward network

#### [CVPR 2017] 3D Human Pose Estimation from a Single Image via Distance Matrix Regression [[pdf]](https://arxiv.org/abs/1611.09010)
_Francesc Moreno-Noguer_
- uses a 2D-to-3D distance matrix regression after detecting the 2D position of the all body joints

#### [ICCV 2017] Monoc-ular 3d human pose estimation by predicting depth on joints [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Nie_Monocular_3D_Human_ICCV_2017_paper.pdf)
_Bruce Xiaohan Nie, Ping Wei, and Song-Chun Zhu_
- a skeleton-LSTM which learns the depth information from global human skeleton features and a patch-LSTM which utilizes the local image evidence around joint locations

#### [3DV 2017] Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision [[pdf]](https://arxiv.org/abs/1611.09813)
_Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, Christian Theobalt_
- improves the performance of 3D pose estimation through transfer of learned features trained on 2D pose dataset

#### [CVPR 2016] Sparseness Meets Deepness: 3D Human Pose Estimation from Monocular Video [[pdf]](https://arxiv.org/abs/1511.09439)
_Xiaowei Zhou, Menglong Zhu, Spyridon Leonardos, Kosta Derpanis, Kostas Daniilidis_
- realizes 3D pose estimates via an Expectation-Maximization algorithm over the entire video sequence from 2D joint uncertainty maps

#### [CVPR 2015] Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction [[pdf]](http://files.is.tue.mpg.de/black/papers/PosePriorCVPR2015.pdf)
_Ijaz Akhter and Michael J. Black_
- learns a pose-dependent model of joint limits as the prior from a motion capture dataset, which is then used to estimate 3D pose from 2D joint locations as an over-complete dictionary of poses

#### [ICCV 2015] Maximum-Margin Structured Learning with Deep Networks for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1508.06708)
_Sijin Li, Weichen Zhang, Antoni B. Chan_
- proposes the structured-output learning that minimize the cosine distance between the latent image and pose embeddings

#### [CVPR 2014] Robust Estimation of 3D Human Poses from a Single Image [[pdf]](https://arxiv.org/abs/1406.2282)
_Chunyu Wang, Yizhou Wang, Zhouchen Lin, Alan L. Yuille, Wen Gao_
- represents a 3D pose as a linear combination of a sparse set of bases learned from 3D human skeletons

#### [ECCV 2012] Reconstructing 3D Human Pose from 2D Image Landmarks [[pdf]](https://www.ri.cmu.edu/pub_files/2012/10/cameraAndPoseCameraReady.pdf)
_Varun Ramakrishna, Takeo Kanade, Yaser Sheikh_
- recovers 3D locations from 2D landmarks by leveraging a large motion capture corpus as a proxy for visual memory




## human shape estimation
---
#### [ICCV 2019] Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop [[pdf]](https://arxiv.org/abs/1909.12828) [[CODE]](https://www.seas.upenn.edu/~nkolot/projects/spin/)
_Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, Kostas Daniilidis_


#### [ICCV 2019] DenseRaC: Joint 3D Pose and Shape Estimation by Dense Render-and-Compare [[pdf]](https://arxiv.org/abs/1910.00116)
_Yuanlu Xu, Song-Chun Zhu, Tony Tung_

#### [ICCV 2019] A Neural Network for Detailed Human Depth Estimation from a Single Image [[pdf]](https://arxiv.org/abs/1910.01275)
_Sicong Tang, Feitong Tan, Kelvin Cheng, Zhaoyang Li, Siyu Zhu, Ping Tan_

## Action recognition and motion prediction
---

#### [arxiv 2019] Adversarial Attack on Skeleton-based HumanAction Recognition [[pdf]](https://arxiv.org/pdf/1909.06500.pdf)
_Jian Liu, Naveed Akhtar, and Ajmal Mian_
