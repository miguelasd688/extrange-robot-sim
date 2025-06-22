ARG DOCKER_IMAGE
ARG IMAGE_VERSION
ARG IMAGE_SUFFIX

FROM $DOCKER_IMAGE:$IMAGE_VERSION$IMAGE_SUFFIX

#ARG USERNAME
#ARG USER_UID=1000
#ARG USER_GID=$USER_UID

# Create the user
#RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#    && apt-get update \
#    && apt-get install -y sudo \
#    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#    && chmod 0440 /etc/sudoers.d/$USERNAME
# Install additonal packages - add any that you need
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python-is-python3 ssh neovim git nano cmake build-essential zenity libx11-dev libgl1-mesa-glx libegl1-mesa

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python-is-python3 ssh neovim git nano cmake build-essential && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --upgrade pip && \
    pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com && \
    echo "[INFO] Clonando Isaac Lab..." && \
    if [ ! -d "IsaacLab" ]; then \
        git clone https://github.com/isaac-sim/IsaacLab.git; \
    else \
        echo "[INFO] Isaac Lab already exists"; \
    fi && \
    echo "[INFO] Instalando Isaac Lab..." && \
    cd IsaacLab && \
    ./isaaclab.sh --install

ENV SHELL /bin/bash

# [Optional] Set the default user. Omit if you want to keep the default as root.
#USER $USERNAME
CMD ["/bin/bash"]