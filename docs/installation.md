## Installation

This code is based on [ScanRefer](https://github.com/daveredrum/ScanRefer). Please also refer to the ScanRefer setup.

- Install PyTorch:
    ```shell
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
    ```

- Install the necessary packages with `requirements.txt`:
    ```shell
    pip install -r requirements.txt
    ```

- Compile the CUDA modules for the PointNet++ backbone:
    ```shell
    cd lib/pointnet2
    python setup.py install
    ```

Note that this code has been tested with Python 3.8, pytorch 1.6.0, and CUDA 10.2 on Ubuntu 18.04.
