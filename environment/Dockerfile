# hash:sha256:7c889ac4a60e38c9bfc6b014198058c0c1c75fd0ef64aa9a98e62f6c9e319440
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:4.101.2python3.12.4

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.4.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    numpy==2.3.2
