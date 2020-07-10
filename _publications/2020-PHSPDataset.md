---
title: "Polarization Human Pose and Shape Dataset"
collection: publications
permalink: /publication/2020-PHSPDataset
date: 2020-04-15
venue: 'arXiv 2020'
citation: "<b>Shihao Zou</b>, Xinxin Zuo, Yiming Qian, Sen Wang, Chi Xu, Minglun Gong and Li Cheng. arXiv 2020."
---

### Polarization Human Shape and Pose Dataset (PHSPD)
Our home-grown dataset of various human shapes and poses. 
- [[pdf]](https://arxiv.org/abs/2004.14899)
- [[code]](https://github.com/JimmyZou/PolarHumanPoseShapeDataset) 
- [[data (GoogleDrive)]]()
- [[data (OneDrive)]]()

<center><img src="/images/pubilication_image_videos/demo_annotation_shape.gif" width="400"/></center>
*A demo video shows the annotated shape rendered on four types of images (one polarization image and three-view color images).*

---
### Details of PHSPDataset
#### Data Acquisition System (four cameras)
- one polarization camera (resolution 1224 x 1024, 4 channel).
- three  Kinects  V2  in  three  different  views  (each  Kinect  v2  has  a  ToF depth  and  a color camera, resolution depth 512 x 424, color 1920 x 1080).
- each frame consists of one polarization image, three-view color images and three-view depth images.
- around 13 frame-per-second
<center><img src="/images/pubilication_image_videos/camera_config.png" width="400"/><img src="/images/pubilication_image_videos/synchronization.png" width="600"/></center>
*Left figure: our camera configuration. Right Figure: the synchroniztion test result of multi-view camera system.*

