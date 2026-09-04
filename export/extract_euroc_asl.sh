#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 RAW_ROOT EXTRACTED_ROOT [mh|mh_v1|all]" >&2
    echo "Example: $0 /data/euroc/raw /data/euroc/extracted mh_v1" >&2
    exit 2
fi

raw_root=${1%/}
extracted_root=${2%/}
scope=${3:-mh_v1}

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

zip_for_sequence() {
    local sequence=$1
    case "${sequence}" in
        MH_*) printf '%s/machine_hall/%s/%s.zip\n' "${raw_root}" "${sequence}" "${sequence}" ;;
        V1_*) printf '%s/vicon_room1/%s/%s.zip\n' "${raw_root}" "${sequence}" "${sequence}" ;;
        V2_*) printf '%s/vicon_room2/%s/%s.zip\n' "${raw_root}" "${sequence}" "${sequence}" ;;
    esac
}

missing=()
for sequence in "${sequences[@]}"; do
    destination="${extracted_root}/${sequence}"
    if [[ -d "${destination}/mav0/cam0/data" && \
          -f "${destination}/mav0/state_groundtruth_estimate0/data.csv" ]]; then
        echo "Already extracted: ${sequence}"
        continue
    fi

    archive=$(zip_for_sequence "${sequence}")
    if [[ ! -f "${archive}" ]]; then
        missing+=("${archive}")
        continue
    fi

    echo "Extracting ${sequence}"
    mkdir -p "${destination}"
    unzip -q "${archive}" -d "${destination}"
done

if (( ${#missing[@]} > 0 )); then
    echo "Missing archives:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

echo "EuRoC extraction complete: ${extracted_root}"
