import os
import glob
import time
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap
from vggt_slam.ground_truth_pose_source import GroundTruthPoseSource

from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument(
    "--image_list",
    type=str,
    default=None,
    help="Optional text file containing one image path per line",
)
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--headless", action="store_true", help="Skip map visualization")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")

parser.add_argument(
    "--backbone",
    type=str,
    default="vggt",
    choices=["vggt", "da3"],
    help="Reconstruction backbone: original VGGT or Depth Anything 3",
)
parser.add_argument(
    "--da3_model",
    type=str,
    default="depth-anything/DA3-LARGE-1.1",
    help="Hugging Face model name or local DA3 checkpoint",
)
parser.add_argument(
    "--da3_process_res",
    type=int,
    default=504,
    help="DA3 native preprocessing resolution",
)
parser.add_argument(
    "--da3_process_res_method",
    type=str,
    default="upper_bound_resize",
    choices=["upper_bound_resize", "lower_bound_resize"],
    help="DA3 native resize method",
)
parser.add_argument(
    "--da3_ref_view_strategy",
    type=str,
    default="saddle_balanced",
    help="DA3 reference-view selection strategy",
)

gt_group = parser.add_argument_group("exact ground-truth pose factors")
gt_group.add_argument(
    "--gt_pose_file",
    type=str,
    default=None,
    help="TUM trajectory: timestamp tx ty tz qx qy qz qw",
)
gt_group.add_argument(
    "--gt_association_tolerance",
    type=float,
    default=0.02,
    help="Maximum timestamp difference in seconds",
)
gt_group.add_argument(
    "--gt_factor_translation_sigma_m",
    type=float,
    default=0.01,
    help="Translation sigma used to weight exact-GT factors",
)
gt_group.add_argument(
    "--gt_factor_rotation_sigma_deg",
    type=float,
    default=0.1,
    help="Rotation sigma used to weight exact-GT factors",
)


def load_backbone(args, device):
    """Load the selected feed-forward reconstruction backbone."""

    if args.backbone == "vggt":
        print("Loading original VGGT backbone...")

        model = VGGT()
        url = (
            "https://huggingface.co/facebook/"
            "VGGT-1B/resolve/main/model.pt"
        )
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(url)
        )

        model.eval()
        model = model.to(torch.bfloat16)
        model = model.to(device)
        return model

    if args.backbone == "da3":
        print("Loading Depth Anything 3 backbone...")
        print(f"DA3 model: {args.da3_model}")

        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(
            args.da3_model
        )
        model = model.to(device)
        model.eval()
        return model

    raise ValueError(
        f"Unsupported backbone: {args.backbone}"
    )


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    if args.backbone == "da3" and args.max_loops != 0:
        raise ValueError(
            "DA3 currently supports only --max_loops 0. "
            "The original loop-closure inference call is "
            "VGGT-specific."
        )

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        enable_loop_closure=(args.max_loops > 0),
    )

    if args.gt_pose_file is not None:
        pose_source = GroundTruthPoseSource(
            args.gt_pose_file,
            args.gt_association_tolerance,
        )
        solver.configure_metric_factors(
            pose_source,
            args.gt_factor_translation_sigma_m,
            args.gt_factor_rotation_sigma_deg,
        )
        print("Exact-GT metric factors enabled from:", args.gt_pose_file)

    solver.backbone_name = args.backbone
    solver.da3_process_res = args.da3_process_res
    solver.da3_process_res_method = (
        args.da3_process_res_method
    )
    solver.da3_ref_view_strategy = (
        args.da3_ref_view_strategy
    )

    print(
        f"Initializing and loading backbone: "
        f"{args.backbone}"
    )


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = load_backbone(
        args,
        device,
    )

    if args.image_list is not None:
        print(f"Loading image paths from {args.image_list}...")
        with open(args.image_list, "r", encoding="utf-8") as file:
            image_names = [
                line.strip()
                for line in file
                if line.strip() and not line.lstrip().startswith("#")
            ]
    else:
        print(f"Loading images from {args.image_folder}...")
        image_names = [
            path
            for path in glob.glob(os.path.join(args.image_folder, "*"))
            if "depth" not in os.path.basename(path).lower()
            and "txt" not in os.path.basename(path).lower()
            and "db" not in os.path.basename(path).lower()
        ]

    missing_images = [path for path in image_names if not os.path.isfile(path)]
    if missing_images:
        raise FileNotFoundError(
            f"Image does not exist: {missing_images[0]}"
        )
    if not image_names:
        raise ValueError("No input images were found.")

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for backbone calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average backbone time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())
    if solver.metric_factor_manager is not None:
        print("Final metric factor summary:", solver.metric_factor_manager.get_summary())


    if args.run_os:
        # Register the viser object-query panel so the user can search for
        # objects in the viewer in addition to the terminal prompt below.
        import threading
        data_lock = threading.Lock()
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return
            
            if query == "q":
                print("Exiting.")
                return
            
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())


            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    if not args.vis_map and not args.headless:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        if not args.skip_dense_log:
            # Log the full point cloud as one file
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
