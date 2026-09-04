#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 EXTRACTED_ROOT [GPU] [mh|mh_v1|all]" >&2
    echo "Example: $0 /data/euroc/extracted 1 mh_v1" >&2
    echo "Environment: SAVE_DENSE=0|1, SKIP_COMPLETED=0|1, METHODS=baseline,gt|baseline|gt" >&2
    exit 2
fi

dataset_root=${1%/}
gpu=${2:-0}
scope=${3:-mh_v1}
save_dense=${SAVE_DENSE:-0}
skip_completed=${SKIP_COMPLETED:-1}
methods_text=${METHODS:-baseline,gt}
IFS=',' read -r -a methods <<< "${methods_text}"

for method in "${methods[@]}"; do
    if [[ "${method}" != "baseline" && "${method}" != "gt" ]]; then
        echo "Unknown method '${method}'. Use METHODS=baseline,gt, baseline, or gt." >&2
        exit 2
    fi
done

machine_hall=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult)
vicon_room1=(V1_01_easy V1_02_medium V1_03_difficult)
vicon_room2=(V2_01_easy V2_02_medium V2_03_difficult)

case "${scope}" in
    mh)
        sequences=("${machine_hall[@]}")
        ;;
    mh_v1)
        sequences=("${machine_hall[@]}" "${vicon_room1[@]}")
        ;;
    all)
        sequences=("${machine_hall[@]}" "${vicon_room1[@]}" "${vicon_room2[@]}")
        ;;
    *)
        echo "Unknown scope '${scope}'. Use mh, mh_v1, or all." >&2
        exit 2
        ;;
esac

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
missing=()
for sequence in "${sequences[@]}"; do
    sequence_dir="${dataset_root}/${sequence}"
    if [[ ! -d "${sequence_dir}/mav0/cam0/data" || \
          ! -f "${sequence_dir}/mav0/state_groundtruth_estimate0/data.csv" ]]; then
        missing+=("${sequence_dir}")
    fi
done

if (( ${#missing[@]} > 0 )); then
    echo "Missing or incomplete extracted sequences:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

is_complete() {
    local method=$1
    local sequence=$2
    local result_dir
    if [[ "${method}" == "baseline" ]]; then
        result_dir="${project_root}/results/da3_baseline_euroc/${sequence}"
    else
        result_dir="${project_root}/results/da3_gt_euroc/${sequence}"
    fi
    [[ -s "${result_dir}/trajectory.txt" && \
       -s "${result_dir}/ape_se3.txt" && \
       -s "${result_dir}/rpe_translation.txt" && \
       -s "${result_dir}/rpe_rotation_deg.txt" && \
       -s "${result_dir}/run.log" ]]
}

validate_pair() {
    local sequence=$1
    local baseline_dir="${project_root}/results/da3_baseline_euroc/${sequence}"
    local gt_dir="${project_root}/results/da3_gt_euroc/${sequence}"

    cmp -s \
        "${baseline_dir}/prepared/cam0_images.txt" \
        "${gt_dir}/prepared/cam0_images.txt" && \
    cmp -s \
        <(awk '!/^#/ && NF {print $1}' "${baseline_dir}/trajectory.txt") \
        <(awk '!/^#/ && NF {print $1}' "${gt_dir}/trajectory.txt")
}

run_one() {
    local method=$1
    local sequence=$2
    local launcher
    if [[ "${method}" == "baseline" ]]; then
        launcher="${project_root}/export/run_da3_baseline_euroc.sh"
    else
        launcher="${project_root}/export/run_da3_gt_euroc.sh"
    fi

    if [[ "${skip_completed}" == "1" ]] && is_complete "${method}" "${sequence}"; then
        echo "SKIP completed: ${sequence} ${method}"
        return 0
    fi

    echo "START: ${sequence} ${method}"
    SAVE_DENSE="${save_dense}" "${launcher}" "${dataset_root}/${sequence}" "${gpu}"
}

mkdir -p "${project_root}/results/euroc_evaluation"
status_file="${project_root}/results/euroc_evaluation/status.tsv"
printf 'sequence\tmethod\tstatus\n' > "${status_file}"
failures=0

for sequence in "${sequences[@]}"; do
    pair_ok=1
    for method in "${methods[@]}"; do
        if run_one "${method}" "${sequence}"; then
            printf '%s\t%s\tOK\n' "${sequence}" "${method}" >> "${status_file}"
        else
            printf '%s\t%s\tFAILED\n' "${sequence}" "${method}" >> "${status_file}"
            failures=$((failures + 1))
            pair_ok=0
        fi
    done
    if (( pair_ok == 1 )); then
        if validate_pair "${sequence}"; then
            printf '%s\tpair_validation\tOK\n' "${sequence}" >> "${status_file}"
        else
            echo "Baseline/GT frame mismatch: ${sequence}" >&2
            printf '%s\tpair_validation\tFAILED\n' "${sequence}" >> "${status_file}"
            failures=$((failures + 1))
        fi
    fi
done

python3 "${project_root}/scripts/summarize_euroc_evaluation.py" \
    --results_root "${project_root}/results" \
    --output_dir "${project_root}/results/euroc_evaluation" \
    --sequences "${sequences[@]}"

echo "Summary: ${project_root}/results/euroc_evaluation/summary.md"
if (( failures > 0 )); then
    echo "${failures} run(s) failed; rerun the same command to resume." >&2
    exit 1
fi
