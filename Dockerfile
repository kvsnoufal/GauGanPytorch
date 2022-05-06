
FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

# Miniconda install copy-pasted from Miniconda's own Dockerfile reachable 
# at: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

# ENV PATH /opt/conda/bin:$PATH

RUN DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Dubai \
    apt-get update && apt-get install -y 
ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone





WORKDIR /code

# COPY requirements.txt .
RUN python -m pip install pip==20.1.1



RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

####################
RUN pip install tensorboard pandas tqdm matplotlib 
RUN conda env create -f dev_flickr/conda_env.yaml
RUN pip install streamlit-drawable-canvas


# RUN echo "mv /code/stylegan2-ada-pytorch/ /workspaces/styleganv2-pt-art/"
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]







