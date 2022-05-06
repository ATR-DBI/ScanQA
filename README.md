# ScanQA: 3D Question Answering for Spatial Scene Understanding

<p align="center"><img width="540" src="./docs/overview.png"></p>

This is the official repository of our paper [**ScanQA: 3D Question Answering for Spatial Scene Understanding (CVPR 2022)**](https://arxiv.org/abs/2112.10482) by Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoki Kawanabe.
## Abstract
We propose a new 3D spatial understanding task for 3D question answering (3D-QA). In the 3D-QA task, models receive visual information from the entire 3D scene of a rich RGB-D indoor scan and answer given textual questions about the 3D scene.
Unlike the 2D-question answering of visual question answering, the conventional 2D-QA models suffer from problems with spatial understanding of object alignment and directions and fail in object localization from the textual questions in 3D-QA. We propose a baseline model for 3D-QA, called the ScanQA model, which learns a fused descriptor from 3D object proposals and encoded sentence embeddings. This learned descriptor correlates language expressions with the underlying geometric features of the 3D scan and facilitates the regression of 3D bounding boxes to determine the described objects in textual questions. We collected human-edited question-answer pairs with free-form answers grounded in 3D objects in each 3D scene. Our new ScanQA dataset contains over 41k question-answer pairs from 800 indoor scenes obtained from the ScanNet dataset. To the best of our knowledge, ScanQA is the first large-scale effort to perform object-grounded question answering in 3D environments.

## Installation

Please refer to [installation guide](docs/installation.md).

## Dataset

Please refer to [data preparation](docs/dataset.md) for preparing the ScanNet v2 and ScanQA datasets.
## Usage

### Training
- Start training the ScanQA model with RGB values:

  ```shell
  python scripts/train.py --use_color --tag <tag_name>
  ```

  For more training options, please run `scripts/train.py -h`.

### Inference
- Evaluation of trained ScanQA models with the val dataset:

  ```shell
  python scripts/eval.py --folder <folder_name> --qa --force
  ```

  <folder_name> corresponds to the folder under outputs/ with the timestamp + <tag_name>.

- Scoring with the val dataset:

  ```shell
  python scripts/score.py --folder <folder_name>
  ```

- Prediction with the test dataset:

  ```shell
  python scripts/predict.py --folder <folder_name> --test_type test_w_obj (or test_wo_obj)
  ```

  The [ScanQA benchmark](https://eval.ai/web/challenges/challenge-page/1715/overview) is hosted on [EvalAI](https://eval.ai/). 
  Please submit the `outputs/<folder_name>/pred.test_w_obj.json` and `pred.test_wo_obj.json` to this site for the evaluation of the test with and without objects.


## Citation
If you find our work helpful for your research. Please consider citing our paper.
```bibtex
@inproceedings{azuma_2022_CVPR,
  title={ScanQA: 3D Question Answering for Spatial Scene Understanding},
  author={Azuma, Daichi and Miyanishi, Taiki and Kurita, Shuhei and Kawanabe, Motoki},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## Acknowledgements
We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection and [daveredrum/ScanRefer](https://github.com/daveredrum/ScanRefer) for the 3D localization codebase.
<!-- [facebookresearch/votenet](https://github.com/daveredrum/ScanRefer) for the 3D object detection codebase and [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) for the CUDA accelerated PointNet++ implementation. -->

## License
ScanQA is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2022 Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, Motoki Kawanabe
