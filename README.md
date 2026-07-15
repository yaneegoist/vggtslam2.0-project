# VGGT-SLAM 2.0 with VGGT and Depth Anything 3

Dockerized VGGT-SLAM 2.0 with support for two reconstruction backbones:

- `vggt` — original VGGT backbone;
- `da3` — Depth Anything 3 with the standard camera head.

The backbone is selected using the `--backbone` argument.

## Project structure

```text
.
├── Makefile
├── docker-compose.yml
├── docker/
│   └── vggtslam2.Dockerfile
├── approaches/
│   ├── vggtslam2/
│   └── depth-anything-3/
├── data/
├── results/
├── scripts/
└── export/
```

The upstream repository versions are recorded in `UPSTREAM_VERSIONS.txt`.

## Requirements

The host machine must have:

- Linux;
- NVIDIA GPU and driver;
- Docker;
- Docker Compose;
- NVIDIA Container Toolkit;
- internet access for the first model download.

## Build the image

```bash
make build-vggtslam2
```

## Check the environment

Check GPU access and PyTorch:

```bash
make check-vggtslam2-gpu
```

Check the VGGT-SLAM installation:

```bash
make check-vggtslam2-import
```

Check the Depth Anything 3 installation:

```bash
make check-da3-import
```

## Enter the container

Use all available GPUs:

```bash
make run-vggtslam2
```

Use only GPU 0:

```bash
make run-vggtslam2 GPU=0
```

Use only GPU 1:

```bash
make run-vggtslam2 GPU=1 
```

The main directories inside the container are:

```text
/app/vggtslam2
/app/depth-anything-3
/data
/results
/scripts
/export
```

Model checkpoints and framework caches are stored in:

```text
export/cache/
```

## Visualization

Allow Docker containers to access the X server:

```bash
make prepare-terminal-for-visualization
```

Then enter the container:

```bash
make run-vggtslam2 GPU=0 
```

## Prepare the smoke-test data

VGGT-SLAM includes the `office_loop.zip` example sequence.

From the repository root:

```bash
OFFICE_ZIP="$(
    find approaches/vggtslam2 \
        -type f \
        -name office_loop.zip \
        | head -n 1
)"

test -n "$OFFICE_ZIP" || {
    echo "office_loop.zip was not found"
    exit 1
}

rm -rf data/office_loop
mkdir -p data/office_loop

unzip -q \
    "$OFFICE_ZIP" \
    -d data/office_loop
```

The images should be available at:

```text
data/office_loop/office_loop/
```

Check them with:

```bash
find data/office_loop/office_loop \
    -maxdepth 1 \
    -type f \
    | head
```

## Smoke test with the original VGGT backbone

```bash
rm -rf results/smoke_vggt
mkdir -p results/smoke_vggt

GPU=0 docker compose run --rm \
    vggtslam2 -lc '
        python main.py \
            --backbone vggt \
            --image_folder /data/office_loop/office_loop \
            --submap_size 16 \
            --overlapping_window_size 4 \
            --max_loops 0 \
            --min_disparity 50 \
            --conf_threshold 25 \
            --log_results \
            --skip_dense_log \
            --log_path /results/smoke_vggt/poses.txt \
            2>&1 | tee /results/smoke_vggt/run.log
    '
```

Check the output:

```bash
wc -l results/smoke_vggt/poses.txt
head -n 3 results/smoke_vggt/poses.txt
tail -n 30 results/smoke_vggt/run.log
```

## Smoke test with the DA3 backbone

DA3 uses:

- `DA3-LARGE-1.1`;
- native preprocessing at resolution 504;
- `upper_bound_resize`;
- `saddle_balanced` reference-view selection;
- standard camera head;
- no ray-pose.

```bash
rm -rf results/smoke_da3
mkdir -p results/smoke_da3

GPU=0 docker compose run --rm \
    vggtslam2 -lc '
        python main.py \
            --backbone da3 \
            --da3_model depth-anything/DA3-LARGE-1.1 \
            --da3_process_res 504 \
            --da3_process_res_method upper_bound_resize \
            --da3_ref_view_strategy saddle_balanced \
            --image_folder /data/office_loop/office_loop \
            --submap_size 16 \
            --overlapping_window_size 4 \
            --max_loops 0 \
            --min_disparity 50 \
            --conf_threshold 25 \
            --log_results \
            --skip_dense_log \
            --log_path /results/smoke_da3/poses.txt \
            2>&1 | tee /results/smoke_da3/run.log
    '
```

Check the output:

```bash
wc -l results/smoke_da3/poses.txt
head -n 3 results/smoke_da3/poses.txt
tail -n 30 results/smoke_da3/run.log
```

## Running on another image sequence

Place the images in a directory under `data/`.

Example:

```text
data/my_sequence/
├── frame_0001.jpg
├── frame_0002.jpg
└── ...
```

Run with the original VGGT backbone:

```bash
GPU=0 docker compose run --rm \
    vggtslam2 -lc '
        python main.py \
            --backbone vggt \
            --image_folder /data/my_sequence \
            --max_loops 0
    '
```

Run with Depth Anything 3:

```bash
GPU=0 docker compose run --rm \
    vggtslam2 -lc '
        python main.py \
            --backbone da3 \
            --image_folder /data/my_sequence \
            --max_loops 0
    '
```

DA3 currently requires:

```text
--max_loops 0
```

## Notes

- The first VGGT run downloads the VGGT checkpoint.
- The first DA3 run downloads the selected DA3 checkpoint.
- Downloads are cached in `export/cache/`.
- The `DISPLAY variable is not set` warning can be ignored for headless runs.
- The DA3 `gsplat` warning can be ignored for standard SLAM inference.
C
