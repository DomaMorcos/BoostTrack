import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion
from rfdetr import RFDETRBase
import supervision as sv
from abc import ABC, abstractmethod
import os
from PIL import Image

class Detector(ABC):
    @abstractmethod
    def __call__(self, img):
        pass

class YoloDetector(Detector):
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)

    def __call__(self, img):
        results = self.model(img)[0]
        annotations = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Only 'person' class
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                annotations.append(xyxy + [conf])
        return torch.tensor(annotations, dtype=torch.float32) if annotations else torch.zeros((0, 5), dtype=torch.float32)

class FasterRCNNDetector(Detector):
    def __init__(self, model_path):
        anchor_sizes = tuple((int(w),) for w, _ in [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet101', pretrained=False, trainable_layers=3
        )
        self.model = FasterRCNN(
            backbone, num_classes=2, rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_test=4000, rpn_post_nms_top_n_test=400,
            rpn_nms_thresh=0.67, box_score_thresh=0.085, box_nms_thresh=0.5,
            box_detections_per_img=300
        )
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transforms = A.Compose([
            A.Resize(height=640, width=640, always_apply=True),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img_rgb)
        img_tensor = augmented['image'].to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        boxes = predictions['boxes'].cpu()
        scores = predictions['scores'].cpu()
        labels = predictions['labels'].cpu()
        mask = (labels == 1)  # Only pedestrian class
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) > 0:
            scale_x, scale_y = orig_w / 640, orig_h / 640
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            annotations = torch.cat((boxes, scores.unsqueeze(1)), dim=1)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)
        
        return annotations

class RFDETRDetector(Detector):
    def __init__(self, model_path):
        self.model = RFDETRBase(pretrain_weights=model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.to(self.device)
        self.temp_img_path = "/tmp/rfdetr_temp_image.jpg"

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(self.temp_img_path)
        
        with torch.no_grad():
            predictions = self.model.predict(self.temp_img_path)
        
        # print(f"RF-DETR predictions: {predictions}")
        
        if isinstance(predictions, sv.Detections):
            boxes = predictions.xyxy
            scores = predictions.confidence
            labels = predictions.class_id
            
            # print(f"RF-DETR boxes: {boxes}")
            # print(f"RF-DETR scores: {scores}")
            # print(f"RF-DETR labels: {labels}")

            if labels is not None and len(labels) > 0:
                person_mask = labels == 1
                boxes = boxes[person_mask]
                scores = scores[person_mask]
            else:
                boxes = np.array([])
                scores = np.array([])

            if len(boxes) > 0:
                annotations = np.hstack((boxes, scores[:, np.newaxis]))
            else:
                annotations = np.zeros((0, 5), dtype=np.float32)
        else:
            print(f"Unexpected RF-DETR prediction format: {type(predictions)}")
            annotations = np.zeros((0, 5), dtype=np.float32)

        if os.path.exists(self.temp_img_path):
            os.remove(self.temp_img_path)

        return torch.tensor(annotations, dtype=torch.float32)

class EnsembleDetector(Detector):
    def __init__(self, model1, model2, model3, model1_weight=0.35, model2_weight=0.5, model3_weight=0.15, 
                 iou_thresh=0.6, conf_thresh=0.3, small_box_threshold=1024):  # 32x32 = 1024 pixels
        self.model1 = model1  # YOLOv12l
        self.model2 = model2  # YOLOv12x
        self.model3 = model3  # RF-DETR
        self.base_weights = [model1_weight, model2_weight, model3_weight]
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.small_box_threshold = small_box_threshold  # Area threshold for "small" boxes

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        preds = [self.model1(img), self.model2(img), self.model3(img)]
        boxes_list, scores_list, labels_list = [], [], []

        for i, pred in enumerate(preds):
            if len(pred) > 0:
                boxes = pred[:, :4].numpy()
                scores = pred[:, 4].numpy()
                boxes_normalized = boxes / np.array([orig_w, orig_h, orig_w, orig_h])
                labels = np.zeros(len(scores))  # All 'person' (class 0)
                
                # Calculate box areas
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Dynamic weights based on box size
                weights = np.full(len(scores), self.base_weights[i])
                if i == 1:  # YOLOv12x (model2)
                    # Reduce weight for small boxes
                    weights[areas < self.small_box_threshold] = self.base_weights[i] * 0.2  # e.g., 0.5 -> 0.1
                elif i == 0 or i == 2:  # YOLOv12l (model1) and RF-DETR (model3)
                    # Boost weight for small boxes
                    weights[areas < self.small_box_threshold] = self.base_weights[i] * 1.5  # e.g., 0.35 -> 0.525, 0.15 -> 0.225
                
                boxes_list.append(boxes_normalized)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append(np.array([]))
                scores_list.append(np.array([]))
                labels_list.append(np.array([]))

        # Use base weights for WBF (per-box weights aren't directly supported, so we approximate)
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=self.base_weights, 
            iou_thr=self.iou_thresh, skip_box_thr=0.0
        )

        # Filter to 'person' (class 0)
        person_mask = labels == 0
        boxes = boxes[person_mask]
        scores = scores[person_mask]

        # Apply confidence threshold
        conf_mask = scores > self.conf_thresh
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]

        if len(boxes) > 0:
            boxes *= np.array([orig_w, orig_h, orig_w, orig_h])
            # Recalculate areas for final output to debug
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            print(f"Final ensemble boxes: {len(boxes)}, Small boxes (<{self.small_box_threshold}): {np.sum(areas < self.small_box_threshold)}")
            annotations = torch.tensor(np.hstack((boxes, scores[:, np.newaxis])), dtype=torch.float32)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)

        return annotations