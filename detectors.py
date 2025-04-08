import torch
from ultralytics import YOLO
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion
from abc import ABC, abstractmethod

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

class EnsembleDetector(Detector):
    def __init__(self, model1, model2, model1_weight=0.35, model2_weight=0.5, 
                 iou_thresh=0.6, conf_thresh=0.3, small_box_threshold=1024):
        self.model1 = model1  # YOLOv12l
        self.model2 = model2  # YOLOv12x
        self.base_weights = [model1_weight, model2_weight]
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.small_box_threshold = small_box_threshold

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        preds = [self.model1(img), self.model2(img)]
        boxes_list, scores_list, labels_list = [], [], []

        for i, pred in enumerate(preds):
            if len(pred) > 0:
                boxes = pred[:, :4].numpy()
                scores = pred[:, 4].numpy()
                boxes_normalized = boxes / np.array([orig_w, orig_h, orig_w, orig_h])
                labels = np.zeros(len(scores))  # All 'person' (class 0)
                
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                weights = np.full(len(scores), self.base_weights[i])
                if i == 1:  # YOLOv12x
                    weights[areas < self.small_box_threshold] = self.base_weights[i] * 0.2
                elif i == 0:  # YOLOv12l
                    weights[areas < self.small_box_threshold] = self.base_weights[i] * 1.5
                
                boxes_list.append(boxes_normalized)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append(np.array([]))
                scores_list.append(np.array([]))
                labels_list.append(np.array([]))

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=self.base_weights, 
            iou_thr=self.iou_thresh, skip_box_thr=0.0
        )

        person_mask = labels == 0
        boxes = boxes[person_mask]
        scores = scores[person_mask]

        conf_mask = scores > self.conf_thresh
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]

        if len(boxes) > 0:
            boxes *= np.array([orig_w, orig_h, orig_w, orig_h])
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            print(f"Final ensemble boxes: {len(boxes)}, Small boxes (<{self.small_box_threshold}): {np.sum(areas < self.small_box_threshold)}")
            annotations = torch.tensor(np.hstack((boxes, scores[:, np.newaxis])), dtype=torch.float32)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)

        return annotations