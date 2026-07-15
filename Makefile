MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(MKFILE_DIR)
DATA_DIR := $(ROOT_DIR)/data
RESULTS_DIR := $(ROOT_DIR)/results
APPROACHES_DIR := $(ROOT_DIR)/approaches

USER_UID := $(shell id -u)
USER_GID := $(shell id -g)

GPU ?= all

.PHONY: init-dirs \
        build-vggtslam2 \
        run-vggtslam2 \
        prepare-terminal-for-visualization \
        check-vggtslam2-gpu \
        check-vggtslam2-import \
        check-da3-import

init-dirs:
	mkdir -p $(APPROACHES_DIR)
	mkdir -p $(DATA_DIR)
	mkdir -p $(RESULTS_DIR)
	mkdir -p $(ROOT_DIR)/docker
	mkdir -p $(ROOT_DIR)/scripts
	mkdir -p $(ROOT_DIR)/export/cache

build-vggtslam2: init-dirs
	USER_UID=$(USER_UID) USER_GID=$(USER_GID) docker compose build vggtslam2

run-vggtslam2:
	GPU=$(GPU) docker compose run --rm vggtslam2

prepare-terminal-for-visualization:
	xhost +local:docker

check-vggtslam2-gpu:
	GPU=$(GPU) docker compose run --rm vggtslam2 -lc "nvidia-smi && python -c 'import torch; print(\"torch:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"device count:\", torch.cuda.device_count()); print(\"device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'"

check-vggtslam2-import:
	GPU=$(GPU) docker compose run --rm vggtslam2 -lc "pwd && python -c 'import os; import torch; import vggt_slam; print(\"main.py exists:\", os.path.exists(\"/app/vggtslam2/main.py\")); print(\"torch import: OK\"); print(\"vggt_slam import: OK\")'"

check-da3-import:
	GPU=$(GPU) docker compose run --rm vggtslam2 -lc "python -c 'import depth_anything_3; import depth_anything_3.api; print(\"depth_anything_3 import: OK\"); print(\"DA3 API import: OK\")'"
