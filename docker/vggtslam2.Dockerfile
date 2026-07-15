FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG USERNAME=docker_user
ARG USER_UID=1000
ARG USER_GID=1000
ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=vggt-slam

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN conda create -y -n ${ENV_NAME} python=3.11 pip \
    && conda clean -afy

ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 \
    torchvision==0.18.1

COPY approaches/vggtslam2/requirements.txt /tmp/vggtslam2_requirements.txt

RUN pip install --no-cache-dir -r /tmp/vggtslam2_requirements.txt

# Additional dependencies required by the original Depth Anything 3 API.
RUN pip install --no-cache-dir \
    safetensors \
    open3d \
    plyfile \
    natsort \
    moviepy==1.0.3 \
    evo \
    pycolmap

RUN mkdir -p /opt/vggtslam2_third_party \
    && cd /opt/vggtslam2_third_party \
    && git clone --depth 1 https://github.com/Dominic101/salad.git \
    && pip install --no-cache-dir -e ./salad \
    && git clone --depth 1 https://github.com/MIT-SPARK/VGGT_SPARK.git vggt \
    && pip install --no-cache-dir -e ./vggt \
    && git clone --depth 1 https://github.com/facebookresearch/perception_models.git \
    && pip install --no-cache-dir -e ./perception_models \
    && git clone --depth 1 https://github.com/facebookresearch/sam3.git \
    && pip install --no-cache-dir -e ./sam3

RUN mkdir -p /app/vggtslam2 /app/depth-anything-3 /data /results /scripts /export \
    && chown -R ${USERNAME}:${USERNAME} \
        /app \
        /data \
        /results \
        /scripts \
        /export \
        /opt/vggtslam2_third_party

USER ${USERNAME}

WORKDIR /app/vggtslam2

ENV PYTHONPATH=/app/vggtslam2:/app/depth-anything-3/src

ENV TORCH_HOME=/export/cache/torch
ENV HF_HOME=/export/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/export/cache/huggingface/hub
ENV XDG_CACHE_HOME=/export/cache

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USERNAME}/.bashrc \
    && echo "conda activate ${ENV_NAME}" >> /home/${USERNAME}/.bashrc

ENTRYPOINT ["/bin/bash"]
