## Data preparation

Since this code is based on [ScanRefer](https://github.com/daveredrum/ScanRefer), you can use the same 3D features. Please also refer to the ScanRefer data preparation.


1. Download the [ScanQA dataset](https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2?usp=sharing) under `data/qa/`. 

    ### Dataset format
    ```shell
    "scene_id": [ScanNet scene id, e.g. "scene0000_00"],
    "object_id": [ScanNet object ids (corresponds to "objectId" in ScanNet aggregation file), e.g. "[8]"],
    "object_names": [ScanNet object names (corresponds to "label" in ScanNet aggregation file), e.g. ["cabinet"]],
    "question_id": [...],
    "question": [...],
    "answers": [...],
    ```

2. Download the preprocessed [GLoVE embedding](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command:
    ```shell
    cd data/scannet/
    python batch_load_scannet_data.py
    ```
<!-- 5. (Optional) Download the preprocessed [multiview features (~36GB)](http://kaldir.vc.in.tum.de/enet_feats.hdf5) and put it under `data/scannet/scannet_data/`. -->
5. (Optional) Pre-process the multiview features from ENet. 

    a. Download [the ENet pretrained weights](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and unzip [the extracted ScanNet frames](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip) under `data/`

    c. Change the data paths in `config.py` marked with __TODO__ accordingly.

    d. Extract the ENet features:
    ```shell
    python scripts/compute_multiview_features.py
    ```

    e. Project ENet features from ScanNet frames to point clouds:
    ```shell
    python scripts/project_multiview_features.py --maxpool
    ```

