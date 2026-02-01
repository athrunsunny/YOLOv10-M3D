# YOLOv10-M3D

## Dataset
Download the KITTI dataset from [**KITTI website**](https://www.cvlibs.net/datasets/kitti/index.php)
The directory will be as follows:


    ├── Your_kitti_data
        ├── ImageSets
        ├── testing
            ├── calib
            ├── image_2
        ├── training
            ├── calib
            ├── image_2
            └── label_2

Write the path to "path" in ultralytics/cfg/datasets/KITTI_3D.yaml.


## Train
~~~
python train_3d.py
~~~

## Results
<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
</div>
