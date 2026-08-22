# From Entity Roles to Relation Triplets: A One-Stage Approach to Scene Graph Generation

## Environmental Setup
Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.

$ conda create -n kakaobrain python=3.7
$ torchaudio = 0.13.1 torchsummary = 1.5.1 torchvision = 0.14.1
$ pip install numpy
$ conda install cython scipy
$ pip install pycocotools
$ pip install opencv-python


## For Data

We evaluate R2T on Visual Genome and HICO-DET.

The original datasets are available at:

- Visual Genome: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- HICO-DET: https://umich-ywchao-hico.github.io/

For Visual Genome, our data preparation follows the setup of RelTR:
https://github.com/yrcong/RelTR/blob/main/data/README.md#for-visual-genome

### Visual Genome

```text
BERI
└── data
    └── vg
        ├── rel.json
        ├── test.json
        ├── train.json
        ├── val.json
        └── images


For HICO-DET, our data preparation follows the setup of HOTR:
https://github.com/kakaobrain/hotr#2-hoi-dataset-setup

### HICO-DET

```
BERI
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

The raw datasets are not redistributed in this repository. Please follow the original dataset licenses and download instructions.

## Training (Train R2T on Visual Genome and HICO-DET on a single node with 2 GPUs)

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py 

## Evaluation

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --eval True

## Pretrained Checkpoints

| Dataset | Checkpoint | Main Results |
| --- | --- | --- |
| Visual Genome | [Google Drive](YOUR_VG_CHECKPOINT_LINK) | mR@50: 10.2, hR@50: 14.7, zR@100: 4.4 |
| HICO-DET | [Google Drive](YOUR_HICO_CHECKPOINT_LINK) | Full mAP: 26.6 |

## License

The original code developed in this repository is released under the
Apache License 2.0.

This codebase builds upon and references several existing research
implementations, including [DETR](https://github.com/facebookresearch/detr),
[RelTR](https://github.com/yrcong/RelTR), and [SpeaQ](https://github.com/mlvlab/SpeaQ). Third-party components remain subject to the copyright
and licensing terms of their original authors. Please refer to the
corresponding upstream repositories for details.


