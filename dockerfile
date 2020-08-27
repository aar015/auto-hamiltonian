FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y curl && \
    curl -sL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y \
    python3 \
    python3-pip \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U setuptools pip

RUN python3 -m pip install --no-cache-dir \
    jupyterlab==2.2.0 \
    torch==1.6.0 \
    jupyter-lsp==0.9.0 \
    python-language-server[all]==0.34.1 \
    pandas==1.0.5 \
    matplotlib==3.3.1 \
    ipywidgets==7.5.1 \
    ipycanvas==0.4.7 \
    ipyevents==0.8.0

RUN jupyter labextension install \
    @karosc/jupyterlab_dracula@2.0.3 \
    @krassowski/jupyterlab-lsp@1.1.0 \
    @jupyter-widgets/jupyterlab-manager@2.0.0 \
    ipycanvas@0.4.7 \
    ipyevents@1.8.0

CMD jupyter lab home \
    --no-browser \
    --allow-root \
    --ip=0.0.0.0 \
    --certfile=/ssl/mycert.pem \
    --keyfile=/ssl/mykey.key \
    --NotebookApp.password="sha1:2aa9839bdae1:f645b9baa0344e3e3e2b9df1daf11dd22e3d86db"