# UMOT: A unified framework for long- and short-term association for multi-object tracking


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MOTR](https://github.com/megvii-research/MOTR).

### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n UMOT python=3.7
    conda activate UMOT
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
