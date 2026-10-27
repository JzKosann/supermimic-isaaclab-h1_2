# SuperMimic-H1_2 training / evaluation image: Isaac Lab 2.3.2 (Isaac Sim 5.1) + this extension (editable install).
# Data, models and logs are mounted from the host at run time (see docker-compose.yml); the image contains only the code.
#
#   docker pull nvcr.io/nvidia/isaac-lab:2.3.2
#   docker build -f docker/Dockerfile -t supermimic-h1_2:latest .      # from the repository root

FROM nvcr.io/nvidia/isaac-lab:2.3.2

WORKDIR /workspace/supermimic

COPY . /workspace/supermimic/

# Python in the image is Isaac Sim's /isaac-sim/python.sh; the extension only depends on huggingface_hub, so torch is not reinstalled
RUN /isaac-sim/python.sh -m pip install -e /workspace/supermimic/source/supermimic_h1_2 \
    && /isaac-sim/python.sh -m pip install wandb tensorboard

RUN mkdir -p /workspace/supermimic/logs /workspace/supermimic/datasets /workspace/supermimic/assets/unitree_model

ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["--help"]
