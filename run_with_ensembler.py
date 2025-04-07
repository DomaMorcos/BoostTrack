import os
import shutil
import time
import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from detectors import *

def get_main_args():
    parser = make_parser()
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--no_reid", action="store_true")
    parser.add_argument("--no_cmc", action="store_true")
    parser.add_argument("--s_sim_corr", action="store_true")
    parser.add_argument("--btpp_arg_iou_boost", action="store_true")
    parser.add_argument("--btpp_arg_no_sb", action="store_true")
    parser.add_argument("--btpp_arg_no_vt", action="store_true")
    parser.add_argument("--no_post", action="store_true")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--reid_path", type=str)
    parser.add_argument("--reid_path2", type=str, default=None)
    parser.add_argument("--reid_weight1", type=float, default=0.5)
    parser.add_argument("--reid_weight2", type=float, default=0.5)
    parser.add_argument("--frame_rate", type=int, default=25)

    # Dynamically handle model paths and weights
    args, unknown = parser.parse_known_args()
    detectors_info = {}
    i = 1
    while True:
        path_key = f"--model{i}_path"
        weight_key = f"--model{i}_weight"
        if path_key not in unknown:
            break
        path_idx = unknown.index(path_key)
        weight_idx = unknown.index(weight_key) if weight_key in unknown else -1
        if path_idx + 1 >= len(unknown) or (weight_idx >= 0 and weight_idx + 1 >= len(unknown)):
            break
        path = unknown[path_idx + 1]
        weight = float(unknown[weight_idx + 1]) if weight_idx >= 0 else 1.0 / i  # Default weight if not specified
        detectors_info[f"model{i}"] = {"path": path, "weight": weight}
        i += 1
    
    args.detectors_info = detectors_info
    if not detectors_info:
        raise ValueError("At least one model path must be provided (e.g., --model1_path)")

    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    if args.test_dataset:
        args.result_folder = args.result_folder.replace("-val", "-test")
    return args

def my_data_loader(main_path):
    img_pathes = [os.path.join(main_path, img) for img in os.listdir(main_path)]
    img_pathes = sorted(img_pathes)
    preproc = dataset.ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    for idx, img_path in enumerate(img_pathes[:], 1):
        np_img = cv2.imread(img_path)
        height, width, _ = np_img.shape
        img, target = preproc(np_img, None, (height, width))
        yield ((img.reshape(1, *img.shape), np_img), target, (height, width, torch.tensor(idx), None, ["test"]), None)

def main():
    args = get_main_args()
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset
    GeneralSettings.values['reid_path'] = args.reid_path
    GeneralSettings.values['max_age'] = args.frame_rate

    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr
    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt

    # Initialize detectors dynamically
    detectors = []
    weights = []
    for model_key, info in args.detectors_info.items():
        model_path = info["path"]
        weight = info["weight"]
        # Determine detector type based on file extension or convention
        if model_path.endswith('.pt'):  # Assuming YOLO models
            detectors.append(YoloDetector(model_path))
        elif 'rf-detr' in model_path.lower():  # Heuristic for RF-DETR
            detectors.append(RFDETRDetector(model_path))
        else:  # Default to Faster R-CNN or add more logic
            detectors.append(FasterRCNNDetector(model_path))
        weights.append(weight)
    
    # Normalize weights to sum to 1 (optional, depending on your preference)
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights] if total_weight > 0 else weights

    det = EnsembleDetector(detectors, weights)

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0

    for (img, np_img), _, info, _ in my_data_loader(args.dataset_path):
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []

        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        pred = det(np_img)
        start_time = time.time()

        if pred is None:
            continue
        targets = tracker.update(pred, img, np_img, tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])
        print(f"{len(ids)} ids detected")
        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids, confs))

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)
    tracker.dump_cache()
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    
    if not args.no_post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        interval = 1000
        utils.dti(post_folder_data, post_folder_data, n_dti=interval, n_min=25)

        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

        res_folder = os.path.join(args.result_folder, args.exp_name, "data")
        post_folder_gbi = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "data")
        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(res_folder):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)
            GBInterpolation(path_in=in_path, path_out=out_path2, interval=interval)
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")

if __name__ == "__main__":
    main()