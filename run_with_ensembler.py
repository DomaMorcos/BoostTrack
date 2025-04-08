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
from detectors import YoloDetector, EnsembleDetector

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
    parser.add_argument("--model1_path", type=str, required=True)  # YOLOv12l
    parser.add_argument("--model1_weight", type=float, default=0.35)
    parser.add_argument("--model2_path", type=str, required=True)  # YOLOv12x
    parser.add_argument("--model2_weight", type=float, default=0.5)
    args = parser.parse_args()

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

def draw_boxes(image, detections, color=(0, 255, 0), label=""):
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2, conf = det.tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

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

    # Initialize detectors
    yolo1_det = YoloDetector(args.model1_path)  # YOLOv12l
    yolo2_det = YoloDetector(args.model2_path)  # YOLOv12x

    # Ensemble with two detectors
    det = EnsembleDetector(
        model1=yolo1_det,
        model2=yolo2_det,
        model1_weight=args.model1_weight,
        model2_weight=args.model2_weight,
        conf_thresh=0.3,
        small_box_threshold=800
    )

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0
    vis_folder = os.path.join(args.result_folder, args.exp_name, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

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

        # Visualization every 50 frames
        if frame_id % 50 == 0:
            yolo1_preds = yolo1_det(np_img)
            yolo2_preds = yolo2_det(np_img)
            ensemble_preds = pred

            print(f"YOLO1 detections: {len(yolo1_preds)}")
            print(f"YOLO2 detections: {len(yolo2_preds)}")
            print(f"Ensemble detections: {len(ensemble_preds)}")

            img_yolo1 = draw_boxes(np_img, yolo1_preds, color=(255, 0, 0), label="YOLO1")
            img_yolo2 = draw_boxes(np_img, yolo2_preds, color=(0, 255, 0), label="YOLO2")
            img_ensemble = draw_boxes(np_img, ensemble_preds, color=(255, 255, 0), label="Ensemble")

            cv2.imwrite(os.path.join(vis_folder, f"{video_name}_{frame_id}_yolo1.jpg"), img_yolo1)
            cv2.imwrite(os.path.join(vis_folder, f"{video_name}_{frame_id}_yolo2.jpg"), img_yolo2)
            cv2.imwrite(os.path.join(vis_folder, f"{video_name}_{frame_id}_ensemble.jpg"), img_ensemble)
            print(f"Saved visualizations for frame {frame_id} in {vis_folder}")

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