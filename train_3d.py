from ultralytics import YOLO

if __name__ == '__main__':
    # train and val neead add rect=False
    model = YOLO('yolov10n-3d.yaml').load('yolov10n.pt')

    model.train(data='KITTI_3D.yaml', task='detect3d', epochs=300, batch=4, imgsz=(384, 1280),
                rect=False, workers=0)

    # Validate the model
    metrics = model.val(data='KITTI_3D.yaml', workers=0, rect=False, imgsz=(384, 1280))  # no arguments needed, dataset and settings remembered

    # Predict with the model
    results = model(r"E:\datasets\KITTI3D\training\image_2\000099.png", save=True, rect=False, imgsz=(384, 1280),
                    calib=r'E:\datasets\KITTI3D\training\calib\000099.txt')  # predict on an image

    # KD
    # # 加载剪枝后的模型权重文件
    # teacher_model_path = r"yolov8m-pose-15.pt"
    # student_model_path = "yolov8s-pose-15.pt"

    # model_s = YOLO(student_model_path)
    # model_t = YOLO(teacher_model_path)
     
    # distillation = model_t.model    # None
    # loss_type = "at"   # mgd, cwd, skd, at, atm, pkd, dkd, qf

    # model_s.train(distillation=distillation, loss_type=loss_type, **overrides)



