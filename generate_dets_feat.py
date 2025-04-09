import os
import pickle
import cv2
import numpy as np
import torch
from argparse import ArgumentParser

import dataset
from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.embedding import EmbeddingComputer
from tracker.boost_track import BoostTrack
from detectors import YoloDetector, EnsembleDetector

def make_parser():
    parser = ArgumentParser("Generate Detections and ReID Features for AdapTrack with BoostTrack++ Processing")
    parser.add_argument("--dataset", type=str, default="mot20", help="Dataset name (e.g., mot20)")
    parser.add_argument("--exp_name", type=str, default="BTPP", help="Experiment name")
    parser.add_argument("--result_folder", type=str, default="results/trackers/", help="Folder to save results")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate of the sequence")
    parser.add_argument("--reid_path", type=str, required=True, help="Path to ReID model weights")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to image sequence (e.g., MOT20-01/img1)")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to first YOLO model weights")
    parser.add_argument("--model1_weight", type=float, default=0.5, help="Weight for first model in ensemble")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to second YOLO model weights")
    parser.add_argument("--model2_weight", type=float, default=0.5, help="Weight for second model in ensemble")
    parser.add_argument("--iou_thresh", type=float, default=0.6, help="IoU threshold for WBF")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold for detections post-WBF")
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
        img, _ = preproc(np_img, None, (height, width))
        img = img.reshape(1, *img.shape)
        print(f"Frame {idx}: Input image shape to detector: {img.shape}")
        yield idx, img, np_img, (height, width, idx, None, ["test"])

def main():
    args = make_parser().parse_args()

    # Set GeneralSettings to match BoostTrack++ configuration
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = True  # Enable ReID
    GeneralSettings.values['reid_path'] = args.reid_path
    GeneralSettings.values['max_age'] = args.frame_rate
    GeneralSettings.values['test_dataset'] = False  # Ensure OSNet is used

    # Initialize detectors
    model1 = YoloDetector(args.model1_path)
    model2 = YoloDetector(args.model2_path)
    det = EnsembleDetector(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)

    # Initialize EmbeddingComputer for ReID features
    embedder = EmbeddingComputer(GeneralSettings['dataset'], False, True, reid_path=GeneralSettings['reid_path'])

    # Initialize BoostTrack to apply confidence boosting
    tracker = None
    det_results = {}
    for frame_id, img, np_img, info in my_data_loader(args.dataset_path):
        if np_img is None:
            det_results[frame_id] = np.zeros((0, 5 + 256), dtype=np.float32)  # OSNet embedding_dim=256
            continue

        print(f"Processing frame {frame_id}\r", end="")
        video_name = info[4][0].split("/")[0]
        tag = f"{video_name}:{frame_id}"

        if frame_id == 1:
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        # Run detection
        pred = det(img)
        if pred is None:
            det_results[frame_id] = np.zeros((0, 5 + 256), dtype=np.float32)
            continue

        # Rescale detections to original image size
        scale = min(img.shape[2] / np_img.shape[0], img.shape[3] / np_img.shape[1])
        pred = pred.cpu().numpy()
        pred[:, :4] /= scale

        # Apply BoostTrack++'s confidence boosting and thresholding
        dets = tracker.process_detections(pred, img, np_img, tag)
        if dets.shape[0] == 0:
            det_results[frame_id] = np.zeros((0, 5 + 256), dtype=np.float32)
            continue

        # Compute ReID features
        dets_embs = np.zeros((dets.shape[0], 256), dtype=np.float32)  # OSNet embedding_dim=256
        if dets.size > 0:
            dets_embs = embedder.compute_embedding(np_img, dets[:, :4], tag)
            print(f"Feature dimension for frame {frame_id}: {dets_embs.shape[1]}")  # Debug print

        # Combine detections and features: [x1, y1, x2, y2, conf, feat_1, ..., feat_256]
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