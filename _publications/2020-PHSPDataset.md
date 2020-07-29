---
title: "Polarization Human Pose and Shape Dataset"
collection: publications
permalink: /publication/2020-PHSPDataset
date: 2020-04-15
venue: 'arXiv 2020'
citation: "<b>Shihao Zou</b>, Xinxin Zuo, Yiming Qian, Sen Wang, Chi Xu, Minglun Gong and Li Cheng. arXiv 2020."
---
---
## A quick view of our PHSPDataset
- report [[pdf]](https://arxiv.org/abs/2004.14899)
- how to use [[code]](https://github.com/JimmyZou/PolarHumanPoseShapeDataset) 
- where to obtain [[data (GoogleDrive)]]()
- where to obtain [[data (OneDrive)]]()

**This dataset can only be used for academic purpose. Commercial use is strictly prohibited without permission.**

If you are interested in using PHSPDataset, please cite the following papers. Thank you.
```
@inproceedings{zou2020detailed,  
  title={3D Human Shape Reconstruction from a Polarization Image},  
  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},  
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
  year={2020}  
} 
@article{zou2020polarization,  
  title={Polarization Human Shape and Pose Dataset},  
  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},  
  journal={arXiv preprint arXiv:2004.14899},  
  year={2020}  
}  
```
### Our PHSPDataset provides:
- one view polarization image
- three-view Kinects v2 (three-view ToF depth and color images)
- 12 subjects (9 males and 3 females)
- each subject to do 3 different groups of actions (18 different actions in total) for 4 times plus one free-style group. (around 22K frames of each subject with about 13 fps)
- annotations of **shape and pose** and video clips of 34 types of **actions**

<center><img src="/images/pubilication_image_videos/eccv2020/demo_annotation_fig/fig2_1.png" width="800"/></center>
<center><img src="/images/pubilication_image_videos/eccv2020/demo_annotation_fig/fig2_2.png" width="800"/></center>
<center><img src="/images/pubilication_image_videos/eccv2020/demo_annotation_fig/fig2_3.png" width="800"/></center>
<center><img src="/images/pubilication_image_videos/eccv2020/demo_annotation_fig/fig2_4.png" width="800"/></center>
<center><img src="/images/pubilication_image_videos/eccv2020/demo_annotation_fig/fig2_5.png" width="800"/></center>
<center>Demo figures show the annotated shape and 3D pose rendered on four types of images (one polarization image and three-view color images).</center>

<center><img src="/images/pubilication_image_videos/eccv2020/demo_annotation_shape.gif" width="400"/></center>
<center>A demo video shows the annotated shape rendered on four types of images.</center>

---
## Details of PHSPDataset
### Data Acquisition System (four cameras)
- one polarization camera (resolution 1224 x 1024, 4 channel).
- three  Kinects  V2  in  three  different  views  (each  Kinect  v2  has  a  ToF depth  and  a color camera, resolution depth 512 x 424, color 1920 x 1080).
- each frame consists of one polarization image, three-view color images and three-view depth images.
- around 13 fps
<center><img src="/images/pubilication_image_videos/eccv2020/camera_config.png" width="300"/><img src="/images/pubilication_image_videos/eccv2020/synchronization.png" width="450"/></center>
<center>Left figure: our camera configuration. Right Figure: the synchroniztion test result of multi-view camera system.</center>

### Dataset content
- 12 subjects (9 males and 3 females)
- each subject to do 3 different groups of actions (18 different actions in total) for 4 times plus one free-style group. (around 22K frames of each subject with about 13 fps)
- annotations of **shape and pose** and video clips of 34 types of **actions**

| group #        | actions           |
|:-------:|:-------------:|
| 1 |  warming-up, walking, running, jumping, drinking, lifting dumbbells |
| 2 |  sitting, eating, driving, reading, phoning, waiting|
| 3 |  presenting, boxing, posing, throwing, greeting, hugging, shaking hands|

<center>The table displays the actions in each group. Subjects are required to do each group of actions for four times, but the order of the actions each time is random.</center>



| subject # | gender | # of original frames | # of annotated frames | # of discarded frames |
|:-------:|:-----:|:-----:|:----:|:-----:|
|1 | female | 22561 | 22241 | 320 (1.4%)|
|2 | male | 24325 | 24186 | 139 (0.5%)|
|3 | male | 23918 | 23470 | 448 (1.8%)|
|4 | male | 24242 | 23906 | 336 (1.4%)|
|5 | male | 24823 | 23430 | 1393 (5.6%)|
|6 | male | 24032 | 23523 | 509 (2.1%)|
|7 | female | 22598 | 22362 | 236 (1.0%)|
|8 | male | 23965 | 23459 | 506 (2.1%)|
|9 | male | 24712 | 24556 | 156 (0.6%)|
|10 | female | 24040 | 23581 | 459 (1.9%)|
|11 | male | 24303 | 23795 | 508 (2.1%)|
|12 | male | 24355 | 23603 | 752 (3.1%)|
|total | - | 287874 | 282112 | 5762 (2.0%)|

<center>The table shows the detail number of frames for each subject and also the number of frames that have SMPL shape and 3D joint annotations.</center>



| Coarse-grained Label | Fine-grained Label | Number of Motions | Total Number |
|:-------:|:-----:|:----:|:-----:|
| Warm up | Warm\_up\_wristankle<br>Warm\_up\_pectoral<br>Warm\_up\_eblowback<br>Warm\_up\_bodylean\_right\_arm<br>Warm\_up\_bodylean\_left\_arm<br>Warm\_up\_bow\_right<br>Warm\_up\_bow\_left | 25<br>45<br>39<br>25<br>24<br>22<br>21 | 201 |
| Walk | Walk |43 | 43 |
| Run | Run | 44 | 44 |
| Jump | Jump\_handsup<br>Jump\_vertical | 50<br>35 | 86 |
| Drink | Drink\_bottle\_righthand<br>Drink\_bottle\_lefthand<br>Drink\_cup\_righthand<br>Drink\_cup\_lefthand<br>Drink\_both\_hands | 25<br>3<br>9<br>10<br>3<br>4 | 81 |
| Lift\_dumbbell | Lift\_dumbbell\_righthand<br>Lift\_dumbbell\_lefthand<br>Lift\_dumbbell\_bothhands<br>Lift\_dumbbell\_overhead<br>Lift\_dumbbell\_bothhands\_bend\_legs | 41<br>41<br>43<br>39<br>34 | 198 |
| Sit | Sit | 47 | 47 |
| Eat | Eat\_righthand<br>Eat\_lefthand<br>Eat\_pie/burger |27<br>18<br>23 | 68 |
| Turn\_steering\_wheel | Turn\_steering\_wheel | 46 | 46 |
| Phone | Take out phone, call and put back<br>Call with left hand | 19<br>33 | 52 |
| Boxing | Boxing\_left\_right<br>Boxing\_left\_upwards<br>Boxing\_right\_upwards<br>Boxing\_right\_left |21<br>34<br>37<br>24 | 116 |
| Throw | Throw\_right\_hand<br>Throw\_both\_hand | 47<br>34 | 80 |
| **Entire Dataset** | - | - | 1061 |

<center>The table displays the annotations of clips of different actions.</center>

TODO: some baselines to be added.


