#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 EUROC_SEQUENCE_DIR [GPU]" >&2
    echo "Example: $0 /path/to/extracted/MH_01_easy 0" >&2
    exit 2
fi

sequence_dir=${1%/}
gpu=${2:-0}
if [[ ! -d "${sequence_dir}/mav0/cam0/data" ]]; then
    echo "cam0 data directory not found: ${sequence_dir}/mav0/cam0/data" >&2
    exit 1
fi
if [[ ! -f "${sequence_dir}/mav0/state_groundtruth_estimate0/data.csv" ]]; then
    echo "GT file not found in: ${sequence_dir}/mav0" >&2
    exit 1
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sequence=$(basename "${sequence_dir}")
result_dir="${project_root}/results/da3_gt_euroc/${sequence}"
container_result_dir="/results/da3_gt_euroc/${sequence}"
prepared_dir="${container_result_dir}/prepared"
mkdir -p "${result_dir}/prepared"

cd "${project_root}"

prepare_command=(
    python /scripts/prepare_euroc_gt.py
    --sequence_root /dataset/mav0
    --output_dir "${prepared_dir}"
    --container_sequence_root /dataset/mav0
)
printf -v prepare_command_string '%q ' "${prepare_command[@]}"

echo "Preparing exact cam0 GT for ${sequence}"
GPU="${gpu}" docker compose run --rm \
    --volume "${sequence_dir}:/dataset:ro" \
    vggtslam2 -lc "${prepare_command_string}"

run_command=(
    python main.py
    --backbone da3
    --image_list "${prepared_dir}/cam0_images.txt"
    --max_loops 0
    --backend_optimize_every_n_submaps "${BACKEND_OPTIMIZE_EVERY_N_SUBMAPS:-4}"
    --backend_max_iterations "${BACKEND_MAX_ITERATIONS:-25}"
    --headless
    --log_results
    --log_path "${container_result_dir}/trajectory.txt"
    --gt_pose_file "${prepared_dir}/gt_cam0_tum.txt"
    --gt_association_tolerance 0.001
    --gt_factor_translation_sigma_m 0.01
    --gt_factor_rotation_sigma_deg 0.1
    --gt_factor_jacobian "${GT_FACTOR_JACOBIAN:-forward}"
)
if [[ "${SAVE_DENSE:-1}" == "0" ]]; then
    run_command+=(--skip_dense_log)
fi
printf -v run_command_string '%q ' "${run_command[@]}"
printf -v run_log_quoted '%q' "${container_result_dir}/run.log"
run_command_string="set -o pipefail; ${run_command_string}2>&1 | tee ${run_log_quoted}"

echo "Running DA3 + exact GT backend on ${sequence} (GPU ${gpu})"
GPU="${gpu}" docker compose run --rm \
    --volume "${sequence_dir}:/dataset:ro" \
    vggtslam2 -lc "${run_command_string}"

evaluation_command="set -o pipefail; evo_ape tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' -a --t_max_diff 0.001 | tee '${container_result_dir}/ape_se3.txt'; evo_rpe tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' --pose_relation trans_part --t_max_diff 0.001 | tee '${container_result_dir}/rpe_translation.txt'; evo_rpe tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' --pose_relation angle_deg --t_max_diff 0.001 | tee '${container_result_dir}/rpe_rotation_deg.txt'"

echo "Evaluating trajectory without scale correction"
GPU="${gpu}" docker compose run --rm \
    --volume "${sequence_dir}:/dataset:ro" \
    vggtslam2 -lc "${evaluation_command}"

echo "Results: ${result_dir}"
