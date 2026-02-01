# YOLOv10-M3D

## Remarks
This project is reproduce from LeAD-M3D, 3D detection has been implemented, but the depth feature of CM3D has not been implemented yet; I will continue to improve knowledge distillation in the future

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


