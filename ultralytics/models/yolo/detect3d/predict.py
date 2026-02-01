# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np
import torch
import cv2

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops, nms
from ultralytics.data.utils import _load_calibration
from ultralytics.data.augment import LetterBox


class Detection3DPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a 3d detections model.

    This class specializes in processing 3d detections model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO 3d detections model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process 3d detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-3d.pt", source=ASSETS)
        >>> predictor = Detection3DPredictor(overrides=args)
        >>> predictor.predict_cli()
    """
    KITTI_MEAN_DIMS = {
        0: [1.53, 1.63, 3.88],  # Car
        1: [1.73, 0.67, 0.87],  # Pedestrian
        2: [1.70, 0.60, 1.76],  # Cyclist
    }
    mean_dims_tensor = torch.tensor(list(KITTI_MEAN_DIMS.values()), dtype=torch.float32)

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "detect3d"
        if getattr(self.args, 'calib', None):
            self.calib = _load_calibration(self.args.calib)

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        self.ori_shape = [item.shape[:2] for item in im]
        if isinstance(self.imgsz, list):
            self.resize_im = [cv2.resize(item, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR) for item in im]
            scales = [(self.imgsz[0] / h, self.imgsz[1] / w) for h, w in self.ori_shape]
            # norm camera intrinsics
            scale_x = scales[0][1]
            scale_y = scales[0][0]
            self.calib['intrinsics'][0, 0] *= scale_x  # fx
            self.calib['intrinsics'][1, 1] *= scale_y  # fy
            self.calib['intrinsics'][0, 2] *= scale_x  # cx
            self.calib['intrinsics'][1, 2] *= scale_y  # cy

        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )

        return [letterbox(image=x) for x in im]

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None

        # non norm intrinsics
        h = preds[0][:, :, 7:8] / 2.0
        intris = torch.from_numpy(self.calib['P2']).unsqueeze(0).to(preds[0].device)
        fx, fy = intris[:, 0, 0], intris[:, 1, 1]
        cx, cy = intris[:, 0, 2], intris[:, 1, 2]
        tx, ty = intris[:, 0, 3] / -intris[:, 0, 0], intris[:, 1, 3] / -intris[:, 1, 1]

        proj_u = preds[0][:, :, 9:10]
        proj_v = preds[0][:, :, 10:11]
        depth = preds[0][:, :, 11:12]
        x_3d = (proj_u - cx.unsqueeze(-1).unsqueeze(-1)) * depth / fx.unsqueeze(-1).unsqueeze(-1) + tx.unsqueeze(-1).unsqueeze(-1)
        y_3d = (proj_v - cy.unsqueeze(-1).unsqueeze(-1)) * depth / fy.unsqueeze(-1).unsqueeze(-1) + ty.unsqueeze(-1).unsqueeze(-1)
        preds[0][:, :, 9:10] = x_3d
        preds[0][:, :, 10:11] = y_3d + h

        # mean_dims = self.mean_dims_tensor.to(preds[0].device)
        # cls = preds[0][:, :, 5:6].long().squeeze(-1)
        # preds[0][:, :, 6:9] += mean_dims[cls]

        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        orig_imgs = self.resize_im
        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

        pred_3ds = pred[:, 6:]
        result.update(points_3d=pred_3ds)
        return result
