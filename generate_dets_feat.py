import os
import pickle
import cv2
import numpy as np
import torch
from argparse import ArgumentParser
from torchvision.ops import nms

import dataset
from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.embedding import EmbeddingComputer
from tracker.boost_track import BoostTrack
from detectors import YoloDetectorV2, EnsembleDetectorV2

def make_parser():
    parser = ArgumentParser("Generate Detections and ReID Features for AdapTrack with BoostTrack++ Processing")
    parser.add_argument("--dataset", type=str, default="mot20", help="Dataset name (e.g., mot20)")
    parser.add_argument("--exp_name", type=str, default="BTPP", help="Experiment name")
    parser.add_argument("--result_folder", type=str, default="results/trackers/", help="Folder to save results")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate of the sequence")
    parser.add_argument("--reid_path", type=str, required=True, help="Path to ReID model weights")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to image sequence (e.g., MOT20-01/img1)")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to first YOLO model weights (yolo12l)")
    parser.add_argument("--model1_weight", type=float, default=0.6, help="Weight for first model in ensemble (yolo12l)")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to second YOLO model weights (yolo12x)")
    parser.add_argument("--model2_weight", type=float, default=0.4, help="Weight for second model in ensemble (yolo12x)")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for WBF (lowered to retain more detections)")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold for detections post-WBF (set to 0.3 as requested)")
    parser.add_argument("--output_pickle", type=str, default="dets_feat.pickle", help="Name of the output pickle file")
    return parser

def my_data_loader(main_path):
    """Load and preprocess images, mimicking BoostTrack++'s pipeline."""
    img_paths = sorted([os.path.join(main_path, img) for img in os.listdir(main_path) if img.endswith(('.jpg', '.png'))])
    preproc = dataset.ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    for idx, img_path in enumerate(img_paths, 1):
        np_img = cv2.imread(img_path)
        if np_img is None:
            print(f"Warning: Failed to load {img_path}")
            yield idx, None, None, None
            continue
        # Debug: Check raw image dimensions
        print(f"Frame {idx}: Raw image shape: {np_img.shape}")
        height, width, _ = np_img.shape
        if height == 0 or width == 0:
            print(f"Error: Invalid image dimensions for {img_path}: height={height}, width={width}")
            yield idx, None, None, None
            continue
        # Scale image to [0, 1] for YOLO
        img_yolo = np_img / 255.0
        # Convert to PyTorch tensor and reshape to (1, 3, height, width)
        img_yolo = torch.from_numpy(img_yolo).float().permute(2, 0, 1)  # (3, height, width)
        img_yolo = img_yolo.reshape(1, *img_yolo.shape)  # (1, 3, height, width)
        # Move to GPU if available
        if torch.cuda.is_available():
            img_yolo = img_yolo.cuda()
        print(f"Frame {idx}: Input image shape to detector: {img_yolo.shape}")
        # For ReID, apply ValTransform (expects input in [0, 1])
        img_reid, _ = preproc(np_img / 255.0, None, (height, width))
        yield idx, img_yolo, np_img, (height, width, idx, None, ["test"])

def main():
    args = make_parser().parse_args()

    # Set GeneralSettings to match BoostTrack++ configuration
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = True  # Enable ReID
    GeneralSettings.values['reid_path'] = args.reid_path
    GeneralSettings.values['max_age'] = args.frame_rate
    GeneralSettings.values['test_dataset'] = False  # Ensure OSNet is used
    GeneralSettings.values['det_thresh'] = 0.005  # Further lowered for better recall

    # Tune confidence boosting
    BoostTrackSettings.values['dlo_boost_coef'] = 0.7  # Increased to make boosting more aggressive

    # Initialize detectors with native resolutions
    model1 = YoloDetectorV2(args.model1_path, input_size=1280)  # yolo12l at 1280x1280
    model2 = YoloDetectorV2(args.model2_path, input_size=960)   # yolo12x at 960x960
    det = EnsembleDetectorV2(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)

    # Initialize EmbeddingComputer for ReID features
    embedder = EmbeddingComputer(GeneralSettings['dataset'], False, True, reid_path=GeneralSettings['reid_path'])

    # Initialize BoostTrack to apply confidence boosting
    tracker = None
    det_results = {}
    for frame_id, img, np_img, info in my_data_loader(args.dataset_path):
        if np_img is None:
            det_results[frame_id] = np.zeros((0, 5 + 256), dtype=np.float32)  # Adjusted to 256 based on output
            continue

        print(f"Processing frame {frame_id}\r", end="")
        video_name = info[4][0].split("/")[0]
        tag = f"{video_name}:{frame_id}"

        if frame_id == 1:
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        # Run detection for each model separately (for debugging)
        model1_preds = model1(img)
        model2_preds = model2(img)
        print(f"Frame {frame_id}: yolo12l detections: {model1_preds.shape[0]}")
        print(f"Frame {frame_id}: yolo12x detections: {model2_preds.shape[0]}")

        # Run ensemble detection
        pred = det(img)
        if pred is None:
            print(f"Frame {frame_id}: No detections after ensemble")
            det_results[frame_id] = np.zeros((0, 5 + 256), dtype=np.float32)  # Adjusted to 256
            continue

        # Debug: Number of detections after ensemble
        print(f"Frame {frame_id}: Detections after ensemble (WBF): {pred.shape[0]}")

        # Rescale detections to original image size (already handled in YoloDetectorV2)
        dets = pred.cpu()

        # Apply NMS to reduce duplicate detections
        if dets.shape[0] > 0:
            boxes = dets[:, :4]  # [x1, y1, x2, y2]
            scores = dets[:, 4]
            keep = nms(boxes, scores, iou_threshold=0.3)  # Increased to retain more detections
            dets = dets[keep].numpy()
            print(f"Frame {frame_id}: Detections after NMS: {dets.shape[0]}")

        # Manually apply confidence boosting (mimicking BoostTrack.process_detections)
        if dets.shape[0] > 0:
            # Apply DLO confidence boost if enabled
            if BoostTrackSettings['use_dlo_boost']:
                dets = tracker.dlo_confidence_boost(
                    dets,
                    BoostTrackPlusPlusSettings['use_rich_s'],
                    BoostTrackPlusPlusSettings['use_sb'],
                    BoostTrackPlusPlusSettings['use_vt']
                )
            # Apply DUO confidence boost if enabled
            if BoostTrackSettings['use_duo_boost']:
                dets = tracker.duo_confidence_boost(dets)

            # Apply detection threshold
            dets = dets[dets[:, 4] >= GeneralSettings['det_thresh']]
            print(f"Frame {frame_id}: Detections after confidence boosting and thresholding: {dets.shape[0]}")

        # Visualize detections for the first 5 frames
        if frame_id <= 5 and dets.shape[0] > 0:
            vis_img = np_img.copy()
            for bbox in dets:  # Renamed 'det' to 'bbox' to avoid shadowing
                x1, y1, x2, y2, conf = bbox[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(f"frame_{frame_id}_detections.jpg", vis_img)
            print(f"Saved visualization for frame {frame_id}")

        if dets.shape[0] == 0:
            det_results[frame_id] = np.zeros((0, 5 + 256), dtype=np.float32)  # Adjusted to 256
            continue

        # Compute ReID features
        dets_embs = np.zeros((dets.shape[0], 256), dtype=np.float32)  # Adjusted to 256 based on output
        if dets.size > 0:
            dets_embs = embedder.compute_embedding(np_img, dets[:, :4], tag)
            print(f"Feature dimension for frame {frame_id}: {dets_embs.shape[1]}")  # Debug print

        # Combine detections and features
        dets_with_feats = np.concatenate((dets, dets_embs), axis=1)
        det_results[frame_id] = dets_with_feats

    # Prepare output directory
    if args.dataset == "mot20":
        result_folder = os.path.join(args.result_folder, "MOT20-val")
    else:
        raise ValueError("Only MOT20 dataset is supported in this script for now.")
    output_dir = os.path.join(result_folder, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    output_pickle_path = os.path.join(output_dir, args.output_pickle)

    # Save detections and features to pickle file
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDetections and features saved to {output_pickle_path}")

    # Clean up
    if tracker is not None:
        tracker.dump_cache()

if __name__ == "__main__":
    main()