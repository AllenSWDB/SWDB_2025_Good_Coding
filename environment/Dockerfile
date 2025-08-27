# hash:sha256:e0a63e9816e1533bf9969b71f259fd6421933731b83fed7d5e8db4e3a971e48e
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
    matplotlib==3.10.5 \
    numpy==2.3.2 \
    pandas==2.3.1 \
    pynwb==3.1.1 \
    scikit-learn==1.7.1
