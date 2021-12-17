FROM ubuntu:20.04


USER root
WORKDIR /root
COPY assets /assets


RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends locales && \
  sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' -e 's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen && \
  locale-gen
ENV LANG=ru_RU.UTF-8
ENV LANGUAGE=ru_RU:ru
ENV LC_LANG=ru_RU.UTF-8
ENV LC_ALL=ru_RU.UTF-8

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
   echo $TZ > /etc/timezone


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $(cat /assets/docker-system-packages.lst | tr '\n' ' ')

RUN ln -s $(which python3) /usr/local/bin/python


# https://gitlab.com/nvidia/container-images/cuda/-/blob/f6b043dd81b0cd0e568dfaca596b8e843caddcdf/dist/11.1.1/ubuntu20.04-x86_64/base/Dockerfile

RUN curl -sL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
  echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
  DEBIAN_FRONTEND=noninteractive apt-get update

ENV CUDA_VERSION 11.1.1

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-cudart-11-1=11.1.74-1 cuda-compat-11-1 && \
  ln -s /usr/local/cuda-11.1 /usr/local/cuda

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
  echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
  ldconfig

#ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:${LD_LIBRARY_PATH}

ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"

# https://gitlab.com/nvidia/container-images/cuda/-/blob/73b454d1723a6e3ed860aaad48c62d6b73afd80c/dist/11.1.1/ubuntu20.04-x86_64/runtime/Dockerfile

ENV NCCL_VERSION 2.8.4

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-libraries-11-1=11.1.1-1 \
    libnpp-11-1=11.1.2.301-1 cuda-nvtx-11-1=11.1.74-1 libcublas-11-1=11.3.0.106-1 libcusparse-11-1=11.3.0.10-1

RUN apt-mark hold libcublas-11-1 libnccl2

# https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-command-line-tools-11-1 libcublas-11-1 \
    cuda-nvrtc-11-1 libcufft-11-1 libcurand-11-1 libcusolver-11-1 libcusparse-11-1 libcudnn8=8.1.0.77-1+cuda11.2 \
    libfreetype6-dev libhdf5-serial-dev libzmq3-dev

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64

# libnvinfer не устанавливается на Ubuntu 20.04: https://forums.developer.nvidia.com/t/is-it-possible-to-install-tensorrt-7-2-2-on-ubuntu-20-04/164245


RUN python3 --version
RUN python3 -m pip install --upgrade 'pip~=21.1.3' && \
  python3 -m pip --version
RUN python3 -m pip install --upgrade wheel setuptools

# Для Tensorflow
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
  ldconfig

RUN python3 -m pip install --requirement /assets/docker-python-packages.lst

RUN   python3 -m spacy download ru_core_news_lg

RUN mkdir -p /usr/share/fasttext && \
  curl -L -o /tmp/wiki.ru.zip https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.zip && \
  unzip /tmp/wiki.ru.zip -d /usr/share/fasttext && \
  rm -v /tmp/wiki.ru.zip

RUN mkdir /usr/share/natasha && \
   curl -L -o /usr/share/natasha/navec_news_v1_1B_250K_300d_100q.tar https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar

## RUN python3 -m textblob.download_corpora

RUN mkdir -p /usr/share/transformers/huggingface/DeepPavlov
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; \
      AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased').save_pretrained('/usr/share/transformers/huggingface/DeepPavlov/rubert-base-cased'); \
      AutoModel.from_pretrained('DeepPavlov/rubert-base-cased').save_pretrained('/usr/share/transformers/huggingface/DeepPavlov/rubert-base-cased')" ## && \


# https://github.com/csurfer/rake-nltk/issues/30:
# устанавливать надо после NLTK и скачивания датасетов
RUN python3 -m pip install rake_nltk


RUN DEBIAN_FRONTEND=noninteractive apt-get autoremove -y
RUN DEBIAN_FRONTEND=noninteractive apt-get clean -y
RUN python3 -m pip cache purge
RUN rm -rf /root/.cache/torch/transformers
#RUN find /usr/share/nltk_data -type f -name '*.zip' -exec rm -vf '{}' ';'


CMD ["/bin/bash"]
