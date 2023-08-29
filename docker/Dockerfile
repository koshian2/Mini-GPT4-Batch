FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update
ENV TZ=Asia/Tokyo
ENV LANG=en_US.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -yq --no-install-recommends python3-pip \
        python3-dev \
        build-essential \
        git \
        wget \
        vim \
        libopencv-dev \
        tzdata && apt-get upgrade -y && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install -U pip &&\ 
    pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 &&\ 
    pip install --no-cache-dir PyYAML==6.0.1
RUN git clone https://github.com/koshian2/MiniGPT-4
WORKDIR MiniGPT-4

# Download pretrained model
RUN wget -P /minigpt4_ckpt https://github.com/koshian2/MiniGPT-4/releases/download/aligned-models/pretrained_minigpt4.pth &&\
    wget -P /minigpt4_ckpt https://github.com/koshian2/MiniGPT-4/releases/download/aligned-models/pretrained_minigpt4_7b.pth &&\
    wget -P /minigpt4_ckpt https://github.com/koshian2/MiniGPT-4/releases/download/aligned-models/pretrained_minigpt4_llama2_7b.pth

# Conver environments.py -> requirements.txt
COPY build_src/convert_yaml.py .
RUN python convert_yaml.py && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade --no-cache-dir bitsandbytes==0.41.1 accelerate==0.22.0
# update library to fix bugs
COPY build_src/init_image_encoders.py .
RUN python init_image_encoders.py

CMD ["python", "app/batch.py", "--gpu-id", "0"]