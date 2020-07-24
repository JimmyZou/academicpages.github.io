---
title: "3D Human Shape Reconstruction from a Polarization Image"
collection: publications
permalink: /publication/2020-polarization-clothed-human-shape
date: 2020-07-15
venue: 'ECCV 2020'
citation: "<b>Shihao Zou</b>, Xinxin Zuo, Yiming Qian, Sen Wang, Chi Xu, Minglun Gong and Li Cheng. In Proceedings of the 16th European Conference on Computer Vision (ECCV) 2020."
---
---
## Abstract
This paper tackles the problem of estimating 3D body shape of clothed humans from single polarized 2D images, i.e. polarization images. Polarization images are known to be able to capture polarized reflected lights that preserve rich geometric cues of an object, which has motivated its recent applications in reconstructing surface normal of the objects of interest. Inspired by the recent advances in human shape estimation from single color images, in this paper, we attempt at estimating human body shapes by leveraging the geometric cues from single polarization images. A dedicated two-stage deep learning approach, SfP, is proposed: given a polarization image, stage one aims at inferring the fined-detailed body surface normal; stage two gears to reconstruct the 3D body shape of clothing details. Empirical evaluations on a synthetic dataset (SURREAL) as well as a real-world dataset (PHSPD) demonstrate the qualitative and quantitative performance of our approach in estimating human poses and shapes. This indicates polarization camera is a promising alternative to the more conventional color or depth imaging for human shape estimation. Further, normal maps inferred from polarization imaging play a significant role in accurately recovering the body shapes of clothed people. [[PDF]](/files/eccv2020.pdf) [[PHSPDataset]](https://jimmyzou.github.io/publication/2020-PHSPDataset)

---
## Results
<center><img src="/images/pubilication_image_videos/demo_clothed_shape/detailed_1.jpg" width="800"/></center>
<center><img src="/images/pubilication_image_videos/demo_clothed_shape/detailed_2.jpg" width="800"/></center>
<center><img src="/images/pubilication_image_videos/demo_clothed_shape/detailed_14.jpg" width="800"/></center>
<center><img src="/images/pubilication_image_videos/demo_clothed_shape/detailed_5.jpg" width="800"/></center>
<center>Demo figures show the results of 3D human clothed shape reconstruction.</center>

<center><img src="/images/pubilication_image_videos/demo_detailed_shape.gif" width="1500"/></center>
<center>A demo video shows the results of clothed human shape reconstruction. (side view)</center>

---

If you are interested in using PHSPDataset, please cite the following papers. Thanks.
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
