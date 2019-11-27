---
title: 'Paper list of human pose and shape estimation'
date: 2019-11-12
permalink: /posts/2019/11/human-pose-shape-estimation/
tags:
  - paper list
  - human pose and shape
---

# Paper list of human pose and shape estimation
---

## Contents<a name="contents"></a>
 - [2D human pose estimation](#2D-human-pose-estimation)
 - [3D human pose estimation](#3D-human-pose-estimation)
 - [human shape estimation](#human-shape-estimation)
 - [action recognition, motion prediction and synthesis](#action-motion)

---

## Related blog
### [https://github.com/wangzheallen/awesome-human-pose-estimation](https://github.com/wangzheallen/awesome-human-pose-estimation)

## 2D human pose estimation<a name="2D-human-pose-estimation"></a>

### [ICCV 2019] Single-Stage Multi-Person Pose Machines [[pdf]](https://arxiv.org/abs/1908.09220)
_Xuecheng Nie, Jianfeng Zhang, Shuicheng Yan, Jiashi Feng_

[[back to top]](#contents)

---
## 3D human pose estimation<a name="3D-human-pose-estimatio"></a>

### [arxiv 2019] Distill Knowledge from NRSfM for Weakly Supervised 3D Pose Learning [[pdf]](https://arxiv.org/abs/1908.06377)
_Chaoyang Wang, Chen Kong, Simon Lucey_
- We propose to learn a 3D pose estimator by distilling knowledge from Non-Rigid Structure from Motion (NRSfM). The goal of NRSfM is to recover 3D shape S and camera matrix M given the observed 2D projections W.

### [arxiv 2019] Convex Optimisation for Inverse Kinematics [[pdf]](https://arxiv.org/abs/1910.11016)
_Tarun Yenamandra, Florian Bernard, Jiayi Wang, Franziska Mueller, Christian Theobalt_


### [arxiv] Geometric Pose Affordance: 3D Human Pose with Scene Constraints [[pdf]](https://arxiv.org/abs/1905.07718)
_Zhe Wang, Liyan Chen, Shaurya Rathore, Daeyun Shin, Charless Fowlkes_
- In this paper, we explore the hypothesis that strong prior information about scene geometry can be used to improve pose estimation accuracy. To tackle this question empirically, we have assembled a novel Geometric Pose Affordance dataset, consisting of multi-view imagery of people interacting with a variety of rich 3D environments. We utilized a commercial motion capture system to collect goldstandard estimates of pose and construct accurate geometric 3D CAD models of the scene itself.
- **Motion Capture dataset GPA** [[dataset]](https://wangzheallen.github.io/GPA.html)


### [arxiv 2019] Semantic Estimation of 3D Body Shape and Pose using Minimal Cameras [[pdf]](https://arxiv.org/abs/1908.03030) 
_Andrew Gilbert, Matthew Trumble, Adrian Hilton, John Collomosse_

### [ICCV 2019] Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image [[pdf]](https://arxiv.org/abs/1907.11346) [[code]](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)
_Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee_
- The pipeline of the proposed system consists of human detection, absolute 3D human root localization, and root-relative 3D single-person pose estimation modules.
- **Multi-persoon 3D pose estimation**

### [arxiv 2019] XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera [[pdf]](https://arxiv.org/abs/1907.00837) [[code]](https://github.com/mehtadushy/SelecSLS-Pytorch)
_Dushyant Mehta, Oleksandr Sotnychenko, Franziska Mueller, Weipeng Xu, Mohamed Elgharib, Pascal Fua, Hans-Peter Seidel, Helge Rhodin, Gerard Pons-Moll, Christian Theobalt_

### [TPAMI 2019] Feature Boosting Network For 3D Pose Estimation [[pdf]](https://arxiv.org/abs/1901.04877)  
_Jun Liu, Henghui Ding, Amir Shahroudy, Ling-Yu Duan, Xudong Jiang, Gang Wang, Alex C. Kot_
- ![](/images/fig_human_pose_shape_estimation/10.png)
- the Hourglass CNN layers [22] are used to learn the convolutional features, then the feature maps for different joints are fed to the LSTD module with CCG for feature boosting. The boosted feature maps of each joint ($$j$$) are fed to the subsequent CNN layers to generate the 2D heatmap ($$H_j$$). Depth information ($$d_j$$) of each joint is regressed based on the summation of the boosted feature maps and the 2D heatmap representations (the feature maps obtained by this summation are also concatenated and fed to the subsequent sub-network as input for further feature boosting).
- To model the graphical dependencies among different parts, instead of linking the units of the ConvLSTM sequentially as in [36], we arrange and link the units of the ConvLSTM (see Figure 1) in our LSTD module by following the above-mentioned dependency graph (see Figure 2). We call this ConvLSTM design “Graphical ConvLSTM”.


### [ICCV 2019] Learnable Triangulation of Human Pose [[pdf]](https://arxiv.org/abs/1905.05754) [[code]](https://saic-violet.github.io/learnable-triangulation/)
_Karim Iskakov, Egor Burkov, Victor Lempitsky, Yury Malkov_
- The first (baseline) solution is a basic differentiable algebraic triangulation with an addition of confidence weights estimated from the input images. The second solution is based on a novel method of volumetric aggregation from intermediate 2D backbone feature maps. The aggregated volume is then refined via 3D convolutions that produce final 3D joint heatmaps and allow modelling a human pose prior.

### [ICCV 2019] Generalizing Monocular 3D Human Pose Estimation in the Wild [[pdf]](https://arxiv.org/abs/1904.05512) [[code]](https://github.com/llcshappy/Monocular-3D-Human-Pose) 
_Luyang Wang, Yan Chen, Zhenhua Guo, Keyuan Qian, Mude Lin, Hongsheng Li, Jimmy S. Ren_
- We propose a principled approach to generate high quality 3D pose ground truth given any in-the-wild image with a person inside. We achieve this by first devising a novel stereo inspired neural network to directly map any 2D pose to high quality 3D counterpart. We then perform a carefully designed geometric searching scheme to further refine the joints. Based on this scheme, we build a large-scale dataset with 400,000 in-the-wild images and their corresponding 3D pose ground truth.
- The existing datasets for 2D human pose estimation such as Leeds Sports Pose dataset (LSP) [13], MPII human pose dataset (MPII) [3] and Ai Challenger dataset for 2D human pose estimation (Ai-Challenger) [43] can be used to extract the high quality 3D labels by the 3D label generator.

### [BMVC 2019] MocapNET: Ensemble of SNN Encoders for 3D Human Pose Estimation in RGB Images [[pdf]](http://users.ics.forth.gr/~argyros/mypapers/2019_09_BMVC_mocapnet.pdf) [[code]](https://github.com/FORTH-ModelBasedTracker/MocapNET)
_Qammaz, Ammar and Argyros, Antonis A_

### [3DV 2019] Multi-Person 3D Human Pose Estimation from Monocular Images [[pdf]](https://arxiv.org/abs/1909.10854)   
_Rishabh Dabral, Nitesh B Gundavarapu, Rahul Mitra, Abhishek Sharma, Ganesh Ramakrishnan, Arjun Jain_
- **Multi-Person 3D Pose Estimation**

### [arxiv 2019] C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion [[pdf]](https://arxiv.org/abs/1909.02533)  
_David Novotny, Nikhila Ravi, Benjamin Graham, Natalia Neverova, Andrea Vedaldi_

### [ICCV 2019] A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image [[pdf]](https://arxiv.org/abs/1908.09999)[[code]](https://github.com/zhangboshen/A2J)   
_Fu Xiong, Boshen Zhang, Yang Xiao, Zhiguo Cao, Taidong Yu, Joey Tianyi Zhou, Junsong Yuan_
- Within A2J, anchor points able to capture global-local spatial context information are densely set on depth image as local regressors for the joints. They contribute to predict the positions of the joints in ensemble way to enhance generalization ability.
- hand dataset: Big Hands 2017, NYU, ICVL. human body: ITOP, K2HPD.

### [ICCV 2019] Optimizing Network Structure for 3D Human Pose Estimation [[pdf]](https://chunyuwang.netlify.com/img/ICCV_2019_CiHai.pdf)  
_Hai Ci, Chunyu Wang, Xiaoxuan Ma, Yizhou Wang_
- In this work, we propose a generic formulation where both GCN and Fully Connected Network (FCN) are its special cases. From this formulation, we discover that GCN has limited representation power when used for estimating 3D poses. We overcome the limitation by introducing Locally Connected Network (LCN) which is naturally implemented by this generic formulation. It notably improves the representation capability over GCN.
- In LCN, each node has a different filter. In addition, the dependence between the joints are specified in a more straightforward and flexible way than GCN.

### [ICCV 2019] Cross View Fusion for 3D Human Pose Estimation [[pdf]](https://chunyuwang.netlify.com/img/ICCV_Cross_view_camera_ready.pdf) [[code]](https://github.com/microsoft/multiview-human-pose-estimation-pytorch)
_Haibo Qiu, Chunyu Wang, Jingdong Wang, Naiyan Wang, Wenjun Zeng_
- First, we introduce a cross-view fusion scheme into CNN to **jointly** estimate 2D poses for multiple views. Consequently, the 2D pose estimation for each view already benefits from other views. Second, we present a **recursive** Pictorial Structure Model to recover the 3D pose from the multi-view 2D poses. It gradually improves the accuracy of 3D pose with affordable computational cost.

### [ICCV 2019] Monocular 3D Human Pose Estimation by Generation and Ordinal Ranking [[pdf]](https://arxiv.org/abs/1904.01324) [[code]](https://github.com/ssfootball04/generative_pose)
_Saurabh Sharma, Pavan Teja Varigonda, Prashast Bindal, Abhishek Sharma, Arjun Jain_
-  We propose a Deep Conditional Variational Autoencoder based model that synthesizes diverse anatomically plausible 3D-pose samples conditioned on the estimated 2D-pose. We propose two strategies for obtaining the final 3D pose- (a) depthordering/ordinal relations to score and weight-average the candidate 3D-poses, referred to as OrdinalScore, and (b) with supervision from an Oracle.

### [CVPR 2019] Unsupervised 3D Pose Estimation with Geometric Self-Supervision [[pdf]](https://arxiv.org/abs/1904.04812) 
_Ching-Hang Chen, Ambrish Tyagi, Amit Agrawal, Dylan Drover, Rohith MV, Stefan Stojanov, James M. Rehg_

### [CVPR 2019] IGE-Net: Inverse Graphics Energy Networksfor Human Pose Estimation and Single-View Reconstruction [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jack_IGE-Net_Inverse_Graphics_Energy_Networks_for_Human_Pose_Estimation_and_CVPR_2019_paper.pdf) 
_Dominic Jack, Frederic Maire, Sareh Shirazi, Anders Eriksson_

### [CVPR 2019] Weakly-Supervised Discovery of Geometry-Aware Representationfor 3D Human Pose Estimation [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.pdf)  
_Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, Liang Lin_
- geometry-aware 3D representation for the human pose to address this limitation by using multiple views in a simple auto-encoder model at the training stage and only 2D keypoint information as supervision.
- ![](/images/fig_human_pose_shape_estimation/11.png)
- ![](/images/fig_human_pose_shape_estimation/12.png)

### [CVPR 2019] Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [[pdf]](https://arxiv.org/abs/1903.02330) [[code]](https://github.com/mkocabas/EpipolarPose)
_Muhammed Kocabas, Salih Karagoz, Emre Akbas_
- ![](/images/fig_human_pose_shape_estimation/13.png)

### [CVPR 2019] On the Continuity of Rotation Representation in Neural Netwworks [[pdf]](https://arxiv.org/pdf/1812.07035.pdf) 
_Yi Zhou*, Connelly Barnes*, Jingwan Lu, Jimei Yang, Hao Li_

### [CVPR 2019] Semantic Graph Convolutional Networks for 3D Human Pose Regression [[pdf]](https://arxiv.org/abs/1904.03345) [[code]](https://github.com/garyzhao/SemGCN) 
_Long Zhao, Xi Peng, Yu Tian, Mubbasir Kapadia, Dimitris N. Metaxas_
- SemGCN learns to capture semantic information such as local and global node relationships, which is not explicitly represented in the graph. These semantic relationships can be learned through end-to-end training from the ground truth without additional supervision or hand-crafted rules.
- **"semantic" is not clear to me.**

### [CVPR 2019] In the Wild Human Pose Estimation Using Explicit 2D Features and Intermediate 3D Representations [[pdf]](https://arxiv.org/abs/1904.03289) 
_Ikhsanul Habibie, Weipeng Xu, Dushyant Mehta, Gerard Pons-Moll, Christian Theobalt_
- It has a network architecture that comprises a new disentangled hidden space encoding of explicit 2D and 3D features, and uses supervision by a new learned projection model from predicted 3D pose. Our algorithm can be jointly trained on image data with 3D labels and image data with only 2D labels.
- ![](/images/fig_human_pose_shape_estimation/14.png)

### [CVPR 2019] RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1902.09868)
_Bastian Wandt, Bodo Rosenhahn_
- One part of the proposed reprojection network (RepNet) learns a mapping from a distribution of 2D poses to a distribution of 3D poses using an adversarial training approach. Another part of the network estimates the camera. This allows for the definition of a network layer that performs the reprojection of the estimated 3D pose back to 2D which results in a reprojection loss function.

### [TPAMI 2019] 3D Human Pose Machines with Self-supervised Learning [[pdf]](https://arxiv.org/pdf/1901.03798.pdf) [[code]](http://www.sysu-hcp.net/3d_pose_ssl/) 
_Keze Wang, Liang Lin, Chenhan Jiang, Chen Qian, and Pengxu Wei_
- ![](/images/fig_human_pose_shape_estimation/15.png)

### [CVPR 2019] Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views [[pdf]](https://arxiv.org/abs/1901.04111) [[code]](https://zju-3dv.github.io/mvpose/)
_Junting Dong, Wen Jiang, Qixing Huang, Hujun Bao, Xiaowei Zhou_
- This paper addresses the problem of 3D pose estimation for multiple people in a few calibrated camera views. The main challenge of this problem is to find the cross-view correspondences among noisy and incomplete 2D pose predictions.
- Our key idea is to use a multi-way matching algorithm to cluster the detected 2D poses in all views. Each resulting cluster encodes 2D poses of the same person across different views and consistent correspondences across the keypoints, from which the 3D pose of each person can be effectively inferred. The proposed convex optimization based multi-way matching algorithm is efficient and robust against missing and false detections, without knowing the number of people in the scene.
- ![](/images/fig_human_pose_shape_estimation/16.png)

### [CVPR 2019] Learning the Depths of Moving People by Watching Frozen People[[pdf]](https://arxiv.org/abs/1904.11111)
_Zhengqi Li, Tali Dekel, Forrester Cole, Richard Tucker, Noah Snavely, Ce Liu, William T. Freeman_

### [CVPR 2019] Monocular Total Capture: Posing Face, Body and Hands in the Wild [[pdf]](https://arxiv.org/abs/1812.01598) [[code]](http://domedb.perception.cs.cmu.edu/monototalcapture.html) 
_Donglai Xiang, Hanbyul Joo, Yaser Sheikh_

### [CVPR 2019] Generating Multiple Hypotheses for 3D Human Pose Estimation with Mixture Density Network [[pdf]](https://arxiv.org/abs/1904.05547) [[code]](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network)  
_Chen Li, Gim Hee Lee_

### [CVPR 2019] Exploiting temporal context for 3d human pose estimation in the wild [[pdf]](https://arxiv.org/abs/1905.04266)
_Anurag Arnab, Carl Doersch, Andrew Zisserman_
- bundle-adjustment-based algorithm for recovering accurate 3D human pose andmeshes from monocular videos

### [ECCV 2018] Integral Human Pose Regression [[pdf]](https://arxiv.org/pdf/1711.08229.pdf) [[code]](https://github.com/JimmySuen/integral-human-pose) 
_Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, Yichen Wei_

### [ECCV 2018] Dense Pose Transfer [[pdf]](https://arxiv.org/pdf/1809.01995.pdf)
_Natalia Neverova, Riza Alp Guler, Iasonas Kokkinos_

### [ECCV 2018] Deep Autoencoder for Combined Human Pose Estimation and Body Model Upscaling [[pdf]](https://cvssp.org/projects/totalcapture/ECCV1UpscalePoseAutoencoder//FinalPaper.pdf)
_Matthew Trumble, Andrew Gilbert, Adrian Hilton, John Collomosse_


### [ECCV 2018] Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Helge_Rhodin_Unsupervised_Geometry-Aware_Representation_ECCV_2018_paper.pdf) [[code]](https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning) 
_Helge Rhodin, Mathieu Salzmann, Pascal Fua_

### [TPAMI 2018] Monocap: Monocular human motion capture using a CNN coupled with a geometric prior [[pdf]](https://arxiv.org/abs/1701.02354) [[code]](https://github.com/daniilidis-group/monocap) 
_Xiaowei Zhou, Menglong Zhu, Georgios Pavlakos, Spyridon Leonardos, Kostantinos G. Derpanis, Kostas Daniilidis_

### [3DV 2018] Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB [[pdf]](https://arxiv.org/pdf/1712.03453.pdf) [[code1]](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) [[code2]](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch)
_Dushyant Mehta, Oleksandr Sotnychenko, Franziska Mueller, Srinath Sridhar, Weipeng Xu,  Gerard Pons-Moll, Christian Theobalt_

### [CVPR 2018] 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning [[pdf]](https://arxiv.org/abs/1802.09232)
_Diogo C. Luvizon, David Picard, Hedi Tabia_
- a multitask framework for jointly 2D and 3D pose estimation from still images and human action recognition from video sequences

### [CVPR 2018] Ordinal Depth Supervision for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1805.04095)
_Georgios Pavlakos, Xiaowei Zhou, Kostas Daniilidis_
- uses ordinal depth relations of human joints for 3D human pose estimation to bypass the need for accurate 3D ground truth

### [ECCV 2018] Exploiting temporal information for 3D pose estimation [[pdf]](https://arxiv.org/abs/1711.08585)
_Mir Rayat Imtiaz Hossain, James J. Little_
- utilizes the temporal information across a sequence of 2D joint locations to estimate a sequence of 3D poses

### [ECCV 2018] Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1804.01110)
_Helge Rhodin, Mathieu Salzmann, Pascal Fua_
- learns a geometry-aware representation using unlabeled multi-view images and then maps to 3D human pose

### [AAAI 2018] Learning Pose Grammar to Encode Human Body Configuration for 3D Pose Estimation [[pdf]](https://arxiv.org/abs/1710.06513)
_Haoshu Fang, Yuanlu Xu, Wenguan Wang, Xiaobai Liu, Song-Chun Zhu_
- proposes a pose grammar to explicitly incorporate a set of knowledge regarding human body configuration as a generalized mapping function from 2D to 3D

### [CVPR 2018] 3D Human Pose Estimation in the Wild by Adversarial Learning [[pdf]](https://arxiv.org/pdf/1803.09722.pdf)
_Wei Yang, Wanli Ouyang, Xiaolong Wang, Jimmy Ren, Hongsheng Li, Xiaogang Wang_

### [CVPR 2018] Ordinal Depth Supervision for 3D Human Pose Estimation [[pdf]](https://arxiv.org/pdf/1805.04095.pdf) [[code]](https://github.com/geopavlakos/ordinal-pose3d/)
_Georgios Pavlakos, Xiaowei Zhou, Kostas Daniilidis_

### [CVPR 2018] V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation From a Single Depth Map [[pdf]](https://arxiv.org/abs/1711.07399) [[code]](https://github.com/mks0601/V2V-PoseNet_RELEASE)
_Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee_

### [IJCAI 2018] DRPose3D: Depth Ranking in 3D Human Pose Estimation [[pdf]](https://arxiv.org/pdf/1805.08973.pdf)
_Min Wang, Xipeng Chen, Wentao Liu, Chen Qian, Liang Lin, Lizhuang Ma_


### [CVPR 2018] Monocular 3D Pose and Shape Estimation of Multiple People in Natural Scenes [[pdf]](http://www.maths.lth.se/sminchisescu/media/papers/Zanfir_Monocular_3D_Pose_CVPR_2018_paper.pdf)
_Andrei Zanfir, Elisabeta Marinoiu, Cristian Sminchisescu_

### [CVPR 2018] Dense Human Pose Estimation In The Wild [[pdf]](https://arxiv.org/pdf/1802.00434.pdf) [[code]](https://github.com/facebookresearch/Densepose)
_Rıza Alp Güler, Natalia Neverova, Iasonas Kokkinos_

### [CVPR 2018] Learning Monocular 3D Human Pose Estimation from Multi-View Images [[pdf]](https://arxiv.org/pdf/1803.04775.pdf)
_Helge Rhodin, Jörg Spörri, Isinsu Katircioglu, Victor Constantin, Frédéric Meyer, Erich Müller, Mathieu Salzmann, Pascal Fua_

### [ECCV 2018] Learning 3D Human Pose from Structure and Motion [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Rishabh_Dabral_Learning_3D_Human_ECCV_2018_paper.pdf)
_Rishabh Dabral, Anurag Mundhada, Uday Kusupati, Safeer Afaque, Abhishek Sharma, Arjun Jain_

### [NeurIPS 2018] Deep Network for the Integrated 3D Sensing ofMultiple People in Natural Images [[pdf]](http://www.maths.lth.se/sminchisescu/media/papers/integrated-3d-sensing-of-multiple-people-in-natural-images_neurips2018.pdf)
_Andrei Zanfir, Elisabeta Marinoiu, Mihai Zanfir, Alin-Ionut Popa, Cristian Sminchisescu_


### [3DV 2017] Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision [[pdf]](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) 
_Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, Christian Theobalt_

### [CVPR 2017] Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose [[pdf]](https://arxiv.org/abs/1611.07828)
_Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis_
- proposes a fine discretization of the 3D space around the subject and a coarse-to-fine prediction scheme to predict per voxel likelihoods for each joint


### [CVPR 2017] A simple yet effective baseline for 3d human pose estimation [[pdf]](https://arxiv.org/abs/1705.03098)
_Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little_

### [TPAMI 2017] A simple, fastand highly-accurate algorithm to recover 3d shape from 2d landmarks on a single image [[pdf]](https://arxiv.org/abs/1609.09058)
_Ruiqi Zhao, Yan Wang, Aleix Martinez_
- “lifting” ground truth 2d joint locations to 3d space is a task that can be solved with a remarkably low error rate with a relatively simple deep feed-forward network

### [CVPR 2017] 3D Human Pose Estimation from a Single Image via Distance Matrix Regression [[pdf]](https://arxiv.org/abs/1611.09010)
_Francesc Moreno-Noguer_
- uses a 2D-to-3D distance matrix regression after detecting the 2D position of the all body joints

### [ICCV 2017] Monocular 3d human pose estimation by predicting depth on joints [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Nie_Monocular_3D_Human_ICCV_2017_paper.pdf)
_Bruce Xiaohan Nie, Ping Wei, and Song-Chun Zhu_
- a skeleton-LSTM which learns the depth information from global human skeleton features and a patch-LSTM which utilizes the local image evidence around joint locations

### [3DV 2017] Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision [[pdf]](https://arxiv.org/abs/1611.09813)
_Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, Christian Theobalt_
- improves the performance of 3D pose estimation through transfer of learned features trained on 2D pose dataset

### [SIGGRAPH 2017] VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera [[pdf]](http://gvv.mpi-inf.mpg.de/projects/VNect/content/VNect_SIGGRAPH2017.pdf) [[code]](https://github.com/timctho/VNect-tensorflow)
_Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko, Helge Rhodin, Mohammad Shafiei, Hans-Peter Seidel, Weipeng Xu, Dan Casas, Christian Theobalt_

### [CVPR 2017] Recurrent 3D Pose Sequence Machines [[pdf]](https://arxiv.org/pdf/1707.09695.pdf) 
_Mude Lin, Liang Lin, Xiaodan Liang, Keze Wang, Hui Cheng_

### [CVPR 2017] Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image [[pdf]](https://arxiv.org/pdf/1701.00295.pdf)
_Denis Tome, Chris Russell, Lourdes Agapito_

### [CVPR 2017] 3D Human Pose Estimation = 2D Pose Estimation + Matching [[pdf]](https://arxiv.org/pdf/1612.06524.pdf) [[code]](https://github.com/flyawaychase/3DHumanPose) 
_Ching-Hang Chen, Deva Ramanan_

### [CVPR 2017] LCR-Net: Localization-Classification-Regression for Human Pose [[pdf]](http://zpascal.net/cvpr2017/Rogez_LCR-Net_Localization-Classification-Regression_for_CVPR_2017_paper.pdf) [[code]](https://thoth.inrialpes.fr/src/LCR-Net/)
_Grégory Rogez, Philippe Weinzaepfel, Cordelia Schmid_


### [CVPR 2017] Harvesting Multiple Views for Marker-less 3D Human Pose Annotations [[pdf]](https://www.seas.upenn.edu/~pavlakos/projects/harvesting/files/harvesting.pdf) [[code]](https://github.com/geopavlakos/harvesting/)
_Georgios Pavlakos, Xiaowei Zhou, Konstantinos G. Derpanis, Kostas Daniilidis_

### [ICCV 2017] Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach [[pdf]](https://arxiv.org/pdf/1704.02447.pdf) [[code]](https://github.com/xingyizhou/Pytorch-pose-hg-3d)
_Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei_

### [TPAMI 2017] Sparse Representation for 3D Shape Estimation: A Convex Relaxation Approach [[pdf]](http://arxiv.org/abs/1509.04309) [[code]](http://www.cad.zju.edu.cn/home/xzhou/code/shapeconvex.zip)
_Xiaowei Zhou, Menglong Zhu, Spyridon Leonardos, Kostas Daniilidis_

### [ICCV 2017] Compositional Human Pose Regression [[pdf]](https://arxiv.org/pdf/1704.00159.pdf)
_Xiao Sun, Jiaxiang Shang, Shuang Liang, Yichen Wei_

### [CVPR 2016] Sparseness Meets Deepness: 3D Human Pose Estimation from Monocular Video [[pdf]](https://arxiv.org/abs/1511.09439)
_Xiaowei Zhou, Menglong Zhu, Spyridon Leonardos, Kosta Derpanis, Kostas Daniilidis_
- realizes 3D pose estimates via an Expectation-Maximization algorithm over the entire video sequence from 2D joint uncertainty maps

### [BMVC 2016] Structured Prediction of 3D Human Pose with Deep Neural Networks [[pdf]](https://arxiv.org/pdf/1605.05180.pdf) 
_Bugra Tekin, Isinsu Katircioglu, Mathieu Salzmann, Vincent Lepetit, Pascal Fua_

### [CVPR 2015] Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction [[pdf]](http://files.is.tue.mpg.de/black/papers/PosePriorCVPR2015.pdf)
_Ijaz Akhter and Michael J. Black_
- learns a pose-dependent model of joint limits as the prior from a motion capture dataset, which is then used to estimate 3D pose from 2D joint locations as an over-complete dictionary of poses

### [ICCV 2015] Maximum-Margin Structured Learning with Deep Networks for 3D Human Pose Estimation [[pdf]](https://arxiv.org/abs/1508.06708)
_Sijin Li, Weichen Zhang, Antoni B. Chan_
- proposes the structured-output learning that minimize the cosine distance between the latent image and pose embeddings

### [CVPR 2014] Robust Estimation of 3D Human Poses from a Single Image [[pdf]](https://arxiv.org/abs/1406.2282)
_Chunyu Wang, Yizhou Wang, Zhouchen Lin, Alan L. Yuille, Wen Gao_
- represents a 3D pose as a linear combination of a sparse set of bases learned from 3D human skeletons

### [ECCV 2012] Reconstructing 3D Human Pose from 2D Image Landmarks [[pdf]](https://www.ri.cmu.edu/pub_files/2012/10/cameraAndPoseCameraReady.pdf)
_Varun Ramakrishna, Takeo Kanade, Yaser Sheikh_
- recovers 3D locations from 2D landmarks by leveraging a large motion capture corpus as a proxy for visual memory


### [CVPR 2012] The vitruvian manifold: Inferring dense correspondences for one-shot human pose estimation [[pdf]](http://www.cs.toronto.edu/~jtaylor/papers/cvpr2012.pdf)
_Jonathan Taylor, Jamie Shotton, Toby Sharp, Andrew Fitzgibbon†_

### [TPAMI 2006] Recovering 3D Human Pose from Monocular Images [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/agarwal-triggs-pami06.pdf)
_Ankur Agarwal and Bill Triggs_

[[back to top]](#contents)

---
## human shape estimation<a name="3D-human-pose-estimation"></a>
### [ICCV 2019] DenseRaC: Joint 3D Pose and Shape Estimation by Dense Render-and-Compare [[pdf]](https://arxiv.org/abs/1910.00116)
_Yuanlu Xu, Song-Chun Zhu, Tony Tung_
- two-step framework takes the body pixel-to-surface correspondence map (i.e., IUV map) as proxy representation and then performs estimation of parameterized human pose and shape.
- construct a large-scale synthetic dataset (MOCA) utilizing web-crawled Mocap sequences, 3D scans and animations. (https://www.mixamo.com/)
- Human Body Model: use a body shape model similar to SMPL. The statistical body model is obtained by PCA on posenormalized 3D models of real humans, obtained by nonrigid registration of a body template to 3D scans of the CAESAR dataset
- ![](/images/fig_human_pose_shape_estimation/2.png)
- comments: multi-person pose and shape estimation, its own human body shape, IUV (pixel-to-surface correspondence) map as input.

### [ICCV 2019] A Neural Network for Detailed Human Depth Estimation from a Single Image (oral) [[pdf]](https://arxiv.org/abs/1910.01275)
_Sicong Tang, Feitong Tan, Kelvin Cheng, Zhaoyang Li, Siyu Zhu, Ping Tan_
- separate the depth map into a smooth base shape and a residual detail shape and design a network with two branches to regress them respectively.
- propose a novel network layer (**non-parametric**) to fuse a rough depth map and surface normals to further improve the final result.
- ![](/images/fig_human_pose_shape_estimation/1.png)
- comments: Reply on the depth information to get the detailed human surface (the amount of this kind of dataset is limited) and the normal estimation part does not actually affect too much. It maybe be better if the human body shape is estimated first and then add the depth details. 

### [ICCV 2019] Moulding Humans: Non-parametric 3D Human Shape Estimation from Single Images [[pdf]](https://arxiv.org/abs/1908.00439)[[studio]](https://kinovis.inria.fr/)
_Valentin Gabeur, Jean-Sebastien Franco, Xavier Martin, Cordelia Schmid, Gregory Rogez_
- propose a non-parametric approach that employs a double depth map to represent the 3D shape of a person: a visible depth map and a “hidden” depth map are estimated and combined, to reconstruct the human 3D shape as done with a “mould”.
- ![](/images/fig_human_pose_shape_estimation/3.png)

### [ICCV 2019] 3DPeople: Modeling the Geometry of Dressed Humans [[pdf]](https://arxiv.org/abs/1904.04571)
_Albert Pumarola, Jordi Sanchez, Gary P. T. Choi, Alberto Sanfeliu, Francesc Moreno-Noguer_

### [ICCV 2019] AMASS: Archive of Motion Capture as Surface Shapes [[pdf]](https://arxiv.org/abs/1904.03278) [[code]](https://github.com/nghorbani/amass) [[code]](https://github.com/nghorbani/amass/tree/master/notebooks)
_Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J._
- use MoSh++ to map this large amount of marker data into our common SMPL pose, shape, and soft-tissue parameters. Due to inherent problems with mocap, such as swapped and mislabeled markers, manually inspect the results and either corrected or held out problems.

### [ICCV 2019] Human Mesh Recovery from Monocular Images via a Skeleton-disentangled Representation [[pdf]](https://arxiv.org/abs/1908.07172) [[code]](https://github.com/Arthur151/DSD-SATN)
_Sun Yu, Ye Yun, Liu Wu, Gao Wenpeng, Fu YiLi, Mei Tao_
- skeleton-disentangling based framework, which divides this task into multi-level spatial and temporal granularity in a decoupling manner. In spatial, we propose an effective and pluggable “disentangling the skeleton from the details” (DSD) module. It reduces the complexity and decouples the skeleton, which lays a good foundation for temporal modeling. In temporal, the selfattention based temporal convolution network is proposed to efficiently exploit the short and long-term temporal cues.
- ![](/images/fig_human_pose_shape_estimation/4.jpg)

### [ICCV 2019] Multi-Garment Net: Learning to Dress 3D People from Images [[pdf]](https://arxiv.org/abs/1908.06903)
_Bharat Lal Bhatnagar, Garvita Tiwari, Christian Theobalt, Gerard Pons-Moll_

### [ICCV 2019] Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild [[pdf]](https://arxiv.org/abs/1908.06442) [[code]](https://penincillin.github.io/dct_iccv2019)
_Yu Rong, Ziwei Liu, Cheng Li, Kaidi Cao, Chen Change Loy_
- perform a comprehensive study on cost and effectiveness trade-off between different annotations.
- obtain several observations: 1) 3D annotations are efficient, whereas traditional 2D annotations such as 2D keypoints and body part segmentation are less competent in guiding 3D human recovery. 2) Dense Correspondence such as DensePose is effective.
- show that incorporating dense correspondence into in the-wild 3D human recovery is promising and competitive due to its high efficiency and relatively low annotating cost
- Interestingly, in the absence of paired 3D data, the models that exploits dense correspondence can achieve 92% of the performance compared to the models trained with paired 3D data (SMPL + 3D joint).

### [ICCV 2019] Shape-Aware Human Pose and Shape Reconstruction Using Multi-View Images [[pdf]](https://arxiv.org/abs/1908.09464)
_Junbang Liang, Ming C. Lin_
- We propose a scalable neural network framework to reconstruct the 3D mesh of a human body from multi-view images, in the subspace of the SMPL model 
- We keep the same cloth textures but apply different background across different views.
- comments: synthetic dataset is not multi-view in reality as background is different.

### [ICCV 2019] Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop [[pdf]](https://arxiv.org/abs/1909.12828) [[code]](https://www.seas.upenn.edu/~nkolot/projects/spin/)
_Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, Kostas Daniilidis_
- A reasonable, directly regressed estimate from the network can initialize the iterative optimization making the fitting faster and more accurate. Similarly, a pixel accurate fit from iterative optimization can act as strong supervision for the network.
- ![](/images/fig_human_pose_shape_estimation/5.png)

### [ICCV 2019] TexturePose: Supervising Human Mesh Estimation with Texture Consistency [[pdf]](https://arxiv.org/pdf/1910.11322.pdf) [[code]](https://github.com/geopavlakos/TexturePose)
_Georgios Pavlakos, Nikos Kolotouros, Kostas Daniilidis_
- We propose a natural form of supervision, that capitalizes on the appearance constancy of a person among different frames (or viewpoints). Assuming that the texture of the person does not change dramatically between frames, we can apply a novel texture consistency loss, which enforces that each point in the texture map has the same texture value across all frames.
- ![](/images/fig_human_pose_shape_estimation/6.png)

### [ICCV 2019] Resolving 3D Human Pose Ambiguities with 3D Scene Constraints [[pdf]](https://arxiv.org/abs/1908.06963) [[Data]](https://prox.is.tue.mpg.de/) [[code]](https://github.com/MohameHassan/prox)
_Mohamed Hassan, Vasileios Choutas, Dimitrios Tzionas, Michael J. Black_
- Exploit static 3D scene structure to better estimate human pose from monocular images. The method enforces Proximal Relationships with Object eXclusion and is called PROX. 
- (No deep learning) The **inter-penetration constraint** penalizes intersection between the body model and the surrounding 3D scene. The **contact constraint** encourages specific parts of the body to be in contact with scene surfaces if they are close enough in distance and orientation. 
- In order to get true ground-truth for the quantitative dataset, we set up a living room in a **marker-based motion capture environment**, scan the scene, and collect RGB-D images in addition to the MoCap data. We fit the SMPL-X model to the MoCap marker data using MoSh++ [41] and this provides ground-truth 3D body shape and pose. This allows us to quantitatively evaluate our method. We reconstruct in total 12 scenes and capture 20 subjects. 

### [CVPR 2019] DeepHuman: 3D Human Reconstruction from a Single Image [[pdf]](https://arxiv.org/abs/1903.06473)
_Zerong Zheng, Tao Yu, Yixuan Wei, Qionghai Dai, Yebin Liu_
- DeepHuman, an image-guided volume-tovolume translation CNN for 3D human reconstruction from a single RGB image. One key feature of our network is that it fuses different scales of image features into the 3D space through volumetric feature transformation, which helps to recover accurate surface geometry. The visible surface details are further refined through a **normal refinement network**, which can be concatenated with the volume generation network using our proposed volumetric normal projection layer (differentiable).

### [CVPR 2019] Learning 3D Human Dynamics from Video [[pdf]](https://arxiv.org/abs/1812.01601) [[code]](https://akanazawa.github.io/human_dynamics/)
_Angjoo Kanazawa, Jason Y. Zhang, Panna Felsen, Jitendra Malik_
- We present a framework that can similarly learn a representation of 3D dynamics of humans from video via a simple but effective temporal encoding of image features.
- we harvest this Internet-scale source of unlabeled data by training our model on unlabeled video with pseudo-ground truth 2D pose obtained from an off-the-shelf 2D pose detector. 
- ![](/images/fig_human_pose_shape_estimation/7.png)

### [CVPR 2019] Detailed Human Shape Estimation from a Single Image by Hierarchical Mesh Deformation [[pdf]](https://arxiv.org/abs/1904.10506) [[code]](https://github.com/zhuhao-nju/hmd) 
_Hao Zhu, Xinxin Zuo, Sen Wang, Xun Cao, Ruigang Yang_
- We propose a novel learningbased framework that combines the robustness of parametric model with the flexibility of free-form 3D deformation. We are able to restore detailed human body shapes beyond skinned models.
- ![](/images/fig_human_pose_shape_estimation/8.png)

### [CVPR 2019] LBS Autoencoder: Self-supervised Fitting of Articulated Meshes to Point Clouds [[pdf]](https://arxiv.org/abs/1904.10037)
_Chun-Liang Li, Tomas Simon, Jason Saragih, Barnabás Póczos, Yaser Sheikh_
- As input, we take a sequence of point clouds to be registered as well as an artist-rigged mesh, i.e. a template mesh equipped with a linear-blend skinning (LBS) deformation space parameterized by a skeleton hierarchy. As output, we learn an LBS-based autoencoder that produces registered meshes from the input point clouds.

### [CVPR 2019] Convolutional Mesh Regression for Single-Image Human Shape Reconstruction (oral) [[pdf]](https://arxiv.org/abs/1905.03244) [[code]](https://www.seas.upenn.edu/~nkolot/projects/cmr/)
_Nikos Kolotouros, Georgios Pavlakos, Kostas Daniilidis_
- ![](/images/fig_human_pose_shape_estimation/9.png)
- Overview of proposed framework. Given an input image, an image-based CNN encodes it in a low dimensional feature vector. This feature vector is embedded in the graph defined by the template human mesh by attaching it to the 3D coordinates $$(x_i, y_i, z_i)$$ of every vertex i. We then process it through a series of Graph Convolutional layers and regress the 3D vertex coordinates $$(\hat x_i, \hat y_i, \hat z_i)$$ of the deformed mesh.

### [CVPR 2019] Expressive Body Capture: 3D Hands, Face, and Body from a Single Image [[pdf]](https://arxiv.org/abs/1904.05866) [[code]](https://smpl-x.is.tue.mpg.de/) [[code]](https://github.com/nghorbani/human_body_prior) [[code]](https://github.com/nghorbani/homogenus)
_Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, Michael J. Black_
- We use thousands of 3D scans to train a new, unified, 3D model of the human body, SMPL-X, that extends SMPL with fully articulated hands and an expressive face. We follow the approach of SMPLify, which estimates 2D features and then optimizes model parameters to fit the features.
- our **PyTorch implementation** achieves a speedup of more than 8× over Chumpy.
- We start from the publicly-available SMPL+H [51] and add the publicly-available FLAME head model [22] to it.

### [CVPR 2019] Volumetric Capture of Humans with a Single RGBD Camera via Semi-Parametric Learning [[pdf]](https://arxiv.org/pdf/1905.12162.pdf)
_Rohit Pandey, Anastasia Tkach, Shuoran Yang, Pavel Pidlypenskyi, Jonathan Taylor, Ricardo Martin-Brualla, Andrea Tagliasacchi, George Papandreou, Philip Davidson, Cem Keskin, Shahram Izadi, Sean Fanello_
- We propose a method to synthesize free viewpoint renderings using a single RGBD camera. The key insight is to leverage previously seen “calibration” images of a given user to extrapolate what should be rendered in a novel viewpoint from the data available in the sensor.
- Given these past observations from multiple viewpoints, and the current RGBD image from a fixed view, we propose an end-to-end framework that fuses both these data sources to generate novel renderings of the performer.

### [arxiv 2019] DenseBody: Directly Regressing Dense 3D Human Pose and Shape From a Single Color Image [[pdf]](https://arxiv.org/pdf/1903.10153.pdf) [[code]](https://github.com/Lotayou/densebody_pytorch)
_Pengfei Yao, Zheng Fang, Fan Wu, Yao Feng, Jiwei Li_

### [arxiv 2019] Towards 3D Human Shape Recovery Under Clothing [[pdf]](https://arxiv.org/abs/1904.02601)
_Xin Chen, Anqi Pang, Yu Zhu, Yuwei Li, Xi Luo, Ge Zhang, Peihao Wang, Yingliang Zhang, Shiying Li, Jingyi Yu_

### [arxiv 2019] Long-Term Video Generation of Multiple FuturesUsing Human Poses [[pdf]](https://arxiv.org/abs/1904.07538)
_Naoya Fushishita, Antonio Tejero-de-Pablos, Yusuke Mukuta, Tatsuya Harada_

### [ICCV 2019] PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization [[pdf]](https://arxiv.org/abs/1905.05172)
_Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, Hao Li_

### [arxiv 2019] Learning 3D Human Body Embedding [[pdf]](https://arxiv.org/abs/1905.05622)
_Boyi Jiang, Juyong Zhang, Jianfei Cai, Jianmin Zheng_

### [arxiv 2019] Shape Evasion: Preventing Body Shape Inference of Multi-Stage Approaches [[pdf]](https://arxiv.org/abs/1905.11503)
_Hosnieh Sattar, Katharina Krombholz, Gerard Pons-Moll, Mario Fritz_

### [arxiv 2019] Temporally Coherent Full 3D Mesh Human Pose Recovery from Monocular Video [[pdf]](https://arxiv.org/abs/1906.00161)
_Jian Liu, Naveed Akhtar, Ajmal Mian_

### [arxiv 2019] Dressing 3D Humans using a Conditional Mesh-VAE-GAN [[pdf]](https://arxiv.org/abs/1907.13615)
_Qianli Ma, Siyu Tang, Sergi Pujades, Gerard Pons-Moll, Anurag Ranjan, Michael J. Black_


### [arxiv 2019] Video Interpolation and Prediction with Unsupervised Landmarks [[pdf]](https://arxiv.org/abs/1909.02749)
_Kevin J. Shih, Aysegul Dundar, Animesh Garg, Robert Pottorf, Andrew Tao, Bryan Catanzaro_

### [3DV 2018] Neural Body Fitting: Unifying Deep Learning and Model-Based Human Pose and Shape Estimation [[pdf]](https://arxiv.org/pdf/1808.05942.pdf) [[code]](https://github.com/mohomran/neural_body_fitting)
_Mohamed Omran, Christoph Lassner, Gerard Pons-Moll, Peter V. Gehler, Bernt Schiele_

### [CVPR 2018] Learning to Estimate 3D Human Pose and Shape from a Single Color Image [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pavlakos_Learning_to_Estimate_CVPR_2018_paper.pdf)
_Georgios Pavlakos, Luyang Zhu, Xiaowei Zhou, Kostas Daniilidis_

### [CVPR 2018] End-to-end Recovery of Human Shape and Pose [[pdf]](https://arxiv.org/pdf/1712.06584.pdf)[[CODE]](https://github.com/akanazawa/hmr)
_Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik_

### [CVPR 2018] Video Based Reconstruction of 3D People Models [[pdf]](https://arxiv.org/pdf/1803.04758.pdf)
_Thiemo Alldieck， Marcus Magnor， Weipeng Xu， Christian Theobalt， Gerard Pons-Moll_

### [SIGGRAPH 2018] Relighting Humans: Occlusion-Aware Inverse Rendering for Full-Body Human Images [[pdf]](https://arxiv.org/abs/1908.02714) [[code]](http://kanamori.cs.tsukuba.ac.jp/projects/relighting_human/)
_Yoshihiro Kanamori, Yuki Endo_

### [ECCV 2018] BodyNet: Volumetric Inference of 3D Human Body Shapes [[pdf]](https://arxiv.org/pdf/1804.04875v3.pdf) [[code]](https://github.com/gulvarol/bodynet)
_Gül Varol, Duygu Ceylan, Bryan Russell, Jimei Yang, Ersin Yumer, Ivan Laptev, Cordelia Schmid_





[[back to top]](#contents)

---
## Action recognition, motion prediction and synthesis<a name="action-motion"></a>

### [arxiv 2019] Learning Variations in Human Motion via Mix-and-Match Perturbation [[pdf]](https://arxiv.org/abs/1908.00733)
_Mohammad Sadegh Aliakbarian, Fatemeh Sadat Saleh, Mathieu Salzmann, Lars Petersson, Stephen Gould, Amirhossein Habibian_

### [arxiv 2019] Peeking into the Future: Predicting Future Person Activities and Locations in Videos [[pdf]](https://arxiv.org/abs/1902.03748)
_Junwei Liang, Lu Jiang, Juan Carlos Niebles, Alexander Hauptmann, Li Fei-Fei_

### [arxiv 2019] Unpaired Pose Guided Human Image Generation [[pdf]](https://arxiv.org/abs/1901.02284)
_Xu Chen, Jie Song, Otmar Hilliges_

### [ICCV 2019] Everybody Dance Now [[pdf]](https://carolineec.github.io/everybody_dance_now/)
_Caroline Chan, Shiry Ginosar, Tinghui Zhou, Alexei A. Efros_

### [ICLR 2018] AUTO-CONDITIONED LSTM NETWORK FOR EXTENDED COMPLEX HUMAN MOTION SYNTHESIS  [[pdf]](https://arxiv.org/pdf/1707.05363.pdf)
_Yi Zhou*, Zimo Li*, Shuangjio Xiao, Chong He, Zeng Huang, Hao Li_


### [AAAI 2019 oral] Joint Dynamic Pose Image and Space Time Reversal for Human Action Recognition from Videos [[pdf]](https://nkliuyifang.github.io/papers/AAAI2019.pdf)
_Mengyuan Liu, Fanyang Meng, Chen Chen, Songtao Wu_

### [TIP 2019] Sample Fusion Network: An End-to-End Data Augmentation Network for Skeleton-based Human Action Recognition [[pdf]](https://nkliuyifang.github.io/papers/TIP2019.pdf)
_Fanyang Meng, Hong Liu, Yongsheng Liang, Juanhui Tu, Mengyuan Liu_

### [arxiv 2019] Context-aware Human Motion Prediction [[pdf]](https://arxiv.org/abs/1904.03419) _Enric Corona, Albert Pumarola, Guillem Alenyà, Francesc Moreno_

### [arxiv 2019] Adversarial Attack on Skeleton-based HumanAction Recognition [[pdf]](https://arxiv.org/pdf/1909.06500.pdf)
_Jian Liu, Naveed Akhtar, and Ajmal Mian_

### [CVPR 2019] Neural Scene Decomposition for Multi-Person Motion Capture [[pdf]](https://arxiv.org/abs/1903.05684)  
_Helge Rhodin, Victor Constantin, Isinsu Katircioglu, Mathieu Salzmann, Pascal Fua_

### [ECCV 2018] Deformable Pose Traversal Convolutionfor 3D Action and Gesture Recognition [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Junwu_Weng_Deformable_Pose_Traversal_ECCV_2018_paper.pdf) 
_Junwu Weng, Mengyuan Liu, Xudong Jiang, Junsong Yuan_

### [NeurIPS 2018] Unsupervised Learning of View-invariant Action Representations [[pdf]](http://papers.nips.cc/paper/7401-unsupervised-learning-of-view-invariant-action-representations.pdf)
_Junnan Li, Yongkang Wong, Qi Zhao, Mohan S. Kankanhalli_

### [NuerIPS 2018] Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis [[pdf]](https://arxiv.org/abs/1810.11610)
_Haoye Dong, Xiaodan Liang, Ke Gong, Hanjiang Lai, Jia Zhu, Jian Yin_

### [CVPR 2017] Deep Learning on Lie Groups for Skeleton-based Action Recognition [[pdf]](https://github.com/zzhiwu/LieNet) [[CODE]](https://github.com/zzhiwu/LieNet) 
_Zhiwu Huang, Chengde Wan, Thomas Probst, Luc Van Gool_

### [ECCV 2018] Few-Shot Human Motion Prediction via Meta-Learning [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangyan_Gui_Few-Shot_Human_Motion_ECCV_2018_paper.pdf)
_Liang-Yan Gui, Yu-Xiong Wang, Deva Ramanan, and Jos_


### [ECCV 2018] MT-VAE: Learning Motion Transformations to Generate Multimodal Human Dynamics [[pdf]](https://arxiv.org/abs/1808.04545)
_Xinchen Yan, Akash Rastogi, Ruben Villegas, Kalyan Sunkavalli, Eli Shechtman, Sunil Hadap, Ersin Yumer, Honglak Lee_

### [CVPR 2018] Deformable GANs for Pose-based Human Image Generation [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Siarohin_Deformable_GANs_for_CVPR_2018_paper.pdf) [[code]](https://github.com/AliaksandrSiarohin/pose-gan)
_Aliaksandr Siarohin, Enver Sangineto, Stephane Lathuiliere, Nicu Sebe_

### [CVPR 2018] A Variational U-Net for Conditional Appearance and Shape Generation [[pdf]](https://compvis.github.io/vunet/images/vunet.pdf) [[code]](https://github.com/CompVis/vunet)
_Patrick Esser, Ekaterina Sutter, Björn Ommer_

[[back to top]](#contents)




