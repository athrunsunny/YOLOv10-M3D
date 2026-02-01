from ultralytics import YOLO

if __name__ == '__main__':
    # 训练验证和推理时需要把rect=False带上
    model = YOLO('yolov10n-3d.yaml').load('yolov10n.pt')

    model.train(data='KITTI_3D.yaml', task='detect3d', epochs=300, batch=4, imgsz=(384, 1280),
                rect=False, workers=0)

    # Validate the model
    metrics = model.val(data='KITTI_3D.yaml', workers=0, rect=False, imgsz=(384, 1280))  # no arguments needed, dataset and settings remembered

    # Predict with the model
    results = model(r"E:\datasets\KITTI3D\training\image_2\000099.png", save=True, rect=False, imgsz=(384, 1280),
                    calib=r'E:\datasets\KITTI3D\training\calib\000099.txt')  # predict on an image


