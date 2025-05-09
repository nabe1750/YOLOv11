# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch
import numpy as np
from PIL import Image
import cv2
import os
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Applies non-max suppression and processes detections.
        construct_results: Constructs a list of result objects from predictions.
        construct_result: Constructs a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor. Defaults to Ultralytics DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
        
        # Depth Anything v2モデルの初期化
        try:
            logger.info("Loading Depth Anything v2 model...")
            self.depth_model = torch.hub.load('LiheYoung/Depth-Anything', 'depth_anything_vitl14', pretrained=True)
            self.depth_model.eval()
            if torch.cuda.is_available():
                self.depth_model = self.depth_model.cuda()
                logger.info("✅ Depth Anything v2 model loaded successfully on GPU")
            else:
                logger.info("✅ Depth Anything v2 model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"❌ Failed to load Depth Anything v2 model: {e}")
            self.depth_model = None

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply non-max suppression and process segmentation detections for each image in the input batch.
        Also performs depth estimation using Depth Anything v2.
        """
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        
        # 深度推定の実行
        depth_maps = []
        if self.depth_model is not None:
            logger.info("Starting depth estimation...")
            for i, orig_img in enumerate(orig_imgs):
                try:
                    if isinstance(orig_img, np.ndarray):
                        orig_img = Image.fromarray(orig_img)
                    depth_map = self._estimate_depth(orig_img)
                    if depth_map is not None:
                        # 深度マップを可視化
                        depth_colormap = cv2.applyColorMap(
                            (depth_map * 255).astype(np.uint8),
                            cv2.COLORMAP_INFERNO
                        )
                        depth_maps.append(depth_colormap)
                    else:
                        depth_maps.append(None)
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{len(orig_imgs)} images for depth estimation")
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    depth_maps.append(None)
        else:
            logger.warning("Depth model not available, skipping depth estimation")
            depth_maps = [None] * len(orig_imgs)
        
        # セグメンテーション結果の後処理
        results = super().postprocess(preds[0], img, orig_imgs, protos=protos)
        
        # 深度マップを各結果に追加
        for result, depth_map in zip(results, depth_maps):
            result.depth_map = depth_map
        
        return results

    def _estimate_depth(self, image):
        """
        Depth Anything v2を使用して深度推定を実行
        """
        try:
            transform = torch.hub.load('LiheYoung/Depth-Anything', 'transform', pretrained=True)
            image = transform(image).unsqueeze(0)
            
            if torch.cuda.is_available():
                image = image.cuda()
            
            with torch.no_grad():
                depth = self.depth_model(image)
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=image.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
                depth = depth.squeeze().cpu().numpy()
            
            return depth
        except Exception as e:
            logger.error(f"Error in depth estimation: {e}")
            return None

    def construct_results(self, preds, img, orig_imgs, protos, depth_maps):
        """
        Construct a list of result objects from the predictions, including depth maps.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto, depth_map)
            for pred, orig_img, img_path, proto, depth_map in zip(preds, orig_imgs, self.batch[0], protos, depth_maps)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto, depth_map):
        """
        Construct a single result object from the prediction, including depth map.
        """
        if not len(pred):  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            pred, masks = pred[keep], masks[keep]
        
        return Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            masks=masks,
            depth_map=depth_map
        )
