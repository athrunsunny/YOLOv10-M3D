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

## Eval 3D 
```python
if __name__ == '__main__':
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect3d.eval import eval_from_scrach

    model = YOLO('yolov10n-3d.pt')
    metrics = model.val(data='KITTI_3D.yaml', workers=0, rect=False, imgsz=(384, 1280), save_txt=True, save_conf=True)
    gt_dir = r'gt labels path'
    det_dir = r'val results path'
    eval_from_scrach(gt_dir, det_dir, eval_cls_list=None, ap_mode=40)
```

## VIZ
<br>
<div align="center">
      <a href="https://github.com/ultralytics"><img src="https:https://github.com/athrunsunny/YOLOv10-M3D/assets/1.jpg" width="3%" alt="Ultralytics GitHub"></a>
</div>

