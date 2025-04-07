import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from abc import ABC, abstractmethod
from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv

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
            if int(box.cls) == 0:  # 'person' class only
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                annotations.append(xyxy + [conf])
        return torch.tensor(annotations, dtype=torch.float32) if annotations else torch.zeros((0, 5), dtype=torch.float32)

class FasterRCNNDetector:
    def __init__(self, model_path):
        anchor_sizes = tuple((int(w),) for w, _ in [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet101', pretrained=False, trainable_layers=3)
        self.model = FasterRCNN(
            backbone, num_classes=2, rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_test=4000, rpn_post_nms_top_n_test=400,
            rpn_nms_thresh=0.67, box_score_thresh=0.085, box_nms_thresh=0.5,
            box_detections_per_img=300)
        
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
        mask = (labels == 1)
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) > 0:
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            annotations = torch.cat((boxes, scores.unsqueeze(1)), dim=1)
        else:
            annotations = torch.zeros((0, 5))
        return annotations

class RFDETRDetector(Detector):
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RFDETRBase(num_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        with torch.no_grad():
            detections = self.model.predict(pil_img, threshold=0.5)
        
        if detections.xyxy.size > 0 and detections.class_id is not None:
            boxes = torch.tensor(detections.xyxy, dtype=torch.float32)
            scores = torch.tensor(detections.confidence, dtype=torch.float32)
            labels = torch.tensor(detections.class_id, dtype=torch.int32)
            
            mask = labels == 0  # Assuming 0 is pedestrian
            boxes = boxes[mask]
            scores = scores[mask]
            
            if boxes.size(0) > 0:
                annotations = torch.cat((boxes, scores.unsqueeze(1)), dim=1)
            else:
                annotations = torch.zeros((0, 5), dtype=torch.float32)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)
        
        return annotations

class EnsembleDetector(Detector):
    def __init__(self, model1_path, model2_path, model3_path, 
                 model1_weight=0.35, model2_weight=0.5, model3_weight=0.15, 
                 iou_thresh=0.6, conf_thresh=0.3):
        # Hardcode Model 1 and 2 as YOLO, Model 3 as RF-DETR
        self.detectors = [
            YoloDetector(model1_path),  # Model 1: YOLO
            YoloDetector(model2_path),  # Model 2: YOLO
            RFDETRDetector(model3_path)  # Model 3: RF-DETR
        ]
        self.weights = [model1_weight, model2_weight, model3_weight]
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]

        # Get predictions from all detectors
        all_preds = [detector(img) for detector in self.detectors]

        # Prepare for WBF
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for preds in all_preds:
            if len(preds) > 0:
                boxes = preds[:, :4].numpy()
                scores = preds[:, 4].numpy()
                boxes_normalized = boxes / np.array([orig_w, orig_h, orig_w, orig_h])
                labels = np.zeros(len(scores))  # Class 0 for 'person'
            else:
                boxes_normalized = np.array([])
                scores = np.array([])
                labels = np.array([])
            
            boxes_list.append(boxes_normalized)
            scores_list.append(scores)
            labels_list.append(labels)

        # Weighted Box Fusion
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=self.weights, 
            iou_thr=self.iou_thresh, skip_box_thr=0.0
        )

        # Filter to 'person' class (0) and apply confidence threshold
        person_mask = labels == 0
        boxes = boxes[person_mask]
        scores = scores[person_mask]
        conf_mask = scores > self.conf_thresh
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]

        # Scale back to original resolution
        if len(boxes) > 0:
            boxes = boxes * np.array([orig_w, orig_h, orig_w, orig_h])
            annotations = torch.tensor(np.hstack((boxes, scores[:, np.newaxis])), dtype=torch.float32)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)

        return annotations