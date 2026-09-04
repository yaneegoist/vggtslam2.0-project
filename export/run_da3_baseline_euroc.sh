#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 EUROC_SEQUENCE_DIR [GPU]" >&2
    echo "Example: $0 /path/to/extracted/MH_01_easy 1" >&2
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
result_dir="${project_root}/results/da3_baseline_euroc/${sequence}"
container_result_dir="/results/da3_baseline_euroc/${sequence}"
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

echo "Preparing the EuRoC image list and evaluation GT for ${sequence}"
GPU="${gpu}" docker compose run --rm \
    --volume "${sequence_dir}:/dataset:ro" \
    vggtslam2 -lc "${prepare_command_string}"

run_command=(
    python main.py
    --backbone da3
    --image_list "${prepared_dir}/cam0_images.txt"
    --max_loops 0
    --headless
    --log_results
    --log_path "${container_result_dir}/trajectory.txt"
)
if [[ "${SAVE_DENSE:-1}" == "0" ]]; then
    run_command+=(--skip_dense_log)
fi
printf -v run_command_string '%q ' "${run_command[@]}"
printf -v run_log_quoted '%q' "${container_result_dir}/run.log"
run_command_string="set -o pipefail; ${run_command_string}2>&1 | tee ${run_log_quoted}"

echo "Running the DA3 + original SL(4) baseline on ${sequence} (GPU ${gpu})"
GPU="${gpu}" docker compose run --rm \
    --volume "${sequence_dir}:/dataset:ro" \
    vggtslam2 -lc "${run_command_string}"

evaluation_command="set -o pipefail; evo_ape tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' -a --t_max_diff 0.001 | tee '${container_result_dir}/ape_se3.txt'; evo_ape tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' -as --t_max_diff 0.001 | tee '${container_result_dir}/ape_sim3.txt'; evo_rpe tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' --pose_relation trans_part --t_max_diff 0.001 | tee '${container_result_dir}/rpe_translation.txt'; evo_rpe tum '${prepared_dir}/gt_cam0_tum.txt' '${container_result_dir}/trajectory.txt' --pose_relation angle_deg --t_max_diff 0.001 | tee '${container_result_dir}/rpe_rotation_deg.txt'"

echo "Evaluating baseline trajectory"
GPU="${gpu}" docker compose run --rm \
    --volume "${sequence_dir}:/dataset:ro" \
    vggtslam2 -lc "${evaluation_command}"

echo "Results: ${result_dir}"
