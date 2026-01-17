# MCC-HO: Multiview Compressive Coding for Hand-Object 3D Reconstruction
Code repository for the paper:
**Reconstructing Hand-Held Objects in 3D from Images and Videos**

[Jane Wu](https://janehwu.github.io/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Georgia Gkioxari](https://gkioxari.github.io/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2404.06507-00ff00.svg)](https://arxiv.org/pdf/2404.06507.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://janehwu.github.io/mcc-ho)

<p align="center">
<img width="1280" alt="teaser" src="https://janehwu.github.io/mcc-ho/mccho_results.png">
</p>

## Quickstart
Pick one of the two demo entrypoints:

- **Without HaMeR (you provide a 3D hand mesh):** `python demo.py ...`
- **With HaMeR (hand is inferred from RGB):** `python demo_with_hamer.py ...` (requires HaMeR + ViTPose/MMPose + Detectron2/OpenMMLab deps)

If you’re just trying to get this repo running end-to-end quickly, start with **Demo (without HaMeR)**. The HaMeR path pulls in OpenMMLab dependencies that are version-sensitive.

## Setup / Installation
Installation and preparation follow [MAE](https://github.com/facebookresearch/mae) / [MCC](https://github.com/facebookresearch/MCC) conventions.

### 0) Clone + initialize submodules
```bash
git clone https://github.com/janehwu/mcc-ho.git
cd mcc-ho
git submodule update --init --recursive
```

### 1) Create a Python environment
There are two realistic choices:

- **If you want HaMeR:** use Python 3.10 (recommended by HaMeR upstream).
- **If you do NOT want HaMeR:** use any Python where PyTorch3D is available.

Example (conda):
```bash
conda create -n mccho python=3.10 -y
conda activate mccho
```

### 2) Install PyTorch (+ CUDA)
Install a PyTorch build matching your CUDA setup. Example (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3) Install PyTorch3D
PyTorch3D is required for demos and training. Follow https://pytorch3d.org/ for the install command matching your PyTorch + CUDA.

### 4) Install MCC-HO Python deps
This repo does not currently ship a pinned `requirements.txt`. These packages are used by the demos/training:
```bash
pip install timm omegaconf trimesh opencv-python matplotlib plotly tqdm
```

### 5) (Optional) Install HaMeR (for `demo_with_hamer.py`)
HaMeR is vendored as a submodule under `third-party/hamer`.

1) Install HaMeR into the *same* environment:
```bash
pip install -e third-party/hamer[all]
```
`third-party/hamer/setup.py` installs several dependencies from GitHub URLs (e.g. Detectron2, chumpy), so this step requires network access and build tools.

2) Install ViTPose/MMPose (used for keypoints):
```bash
pip install -v -e third-party/hamer/third-party/ViTPose
```

3) Download HaMeR demo weights:
```bash
bash third-party/hamer/fetch_demo_data.sh
```

4) Download the MANO right-hand model (`MANO_RIGHT.pkl`) and place it at:
`third-party/hamer/_DATA/data/mano/MANO_RIGHT.pkl`

5) Make sure this repo can find HaMeR’s `_DATA` and ViTPose folders:
```bash
ln -s third-party/hamer/_DATA _DATA
ln -s hamer/third-party/ViTPose third-party/ViTPose
```
Note: the ViTPose symlink target is relative to `third-party/` (so `hamer/third-party/ViTPose` resolves to `third-party/hamer/third-party/ViTPose`).

## Data
Please see [DATASET.md](DATASET.md) for information on data preparation.

This code uses PyTorch3D/Implicitron for data loading and caches metadata to speed up training. To build the cache:
```bash
cd scripts
python prepare_data.py \
  --dataset_path [path to MCC-HO preprocessed data] \
  --dataset_cache [path to cache output folder] \
  --subset [train|val|test]
```

## Demo (without HaMeR)
Run MCC-HO inference on an input image + an input 3D hand mesh:
```bash
python demo.py \
    --image demo/boardgame_v_W_qdSiPKSdQ_frame000019.jpg \
    --hand demo/boardgame_v_W_qdSiPKSdQ_frame000019_hand.obj \
    --seg demo/boardgame_v_W_qdSiPKSdQ_frame000019_mask.png \
    --cam demo/camera_intrinsics_mow.json \
    --checkpoint [path to MCC-HO checkpoint]
```

Notes:
- The camera intrinsics must correspond to the input 3D hand coordinate system.
- Outputs are written under `out_demo/` (HTML visualization + exported point clouds/meshes).

## Demo (with HaMeR)
Run MCC-HO inference on an input image, using HaMeR to infer the 3D hand:
```bash
python demo_with_hamer.py \
  --image demo/drink_v_1F96GArORtg_frame000084.jpg \
  --obj_seg demo/drink_v_1F96GArORtg_frame000084_mask.png \
  --cam demo/camera_intrinsics_hamer.json \
  --checkpoint [path to MCC-HO checkpoint]
```

Notes:
- The object segmentation mask (`obj_seg`) can be obtained using any off-the-shelf segmentation model (e.g. [SAM 2](https://github.com/facebookresearch/sam2)).
- Outputs are written under `out_demo/` (including the preprocessed HaMeR hand at `out_demo/input_hand.obj`).

## Checkpoints
You can use a checkpoint from training (below) or download the pretrained model (trained on DexYCB, MOW, and HOI4D) [[here](https://drive.google.com/file/d/17VOYtywmKhDh_JUULT_M20TNByBUUbqZ/view?usp=sharing)].

## Training
To train an MCC-HO model, please run
```bash
OUTPUT=model_outputs
python main_mccho.py \
    --mccho_path [path to MCC-HO preprocessed data] \
    --dataset_cache [path to dataset cache] \
    --job_dir $OUTPUT \
    --output_dir $OUTPUT/log \
    --shuffle_train
```
- Optional: MCC-HO (excluding the segmentation output layers) may be initialized using MCC pre-trained. A pretrained MCC model is available [[here](https://dl.fbaipublicfiles.com/MCC/co3dv2_all_categories.pth)].

## Troubleshooting
### HaMeR / ViTPose / OpenMMLab issues
- `ModuleNotFoundError: No module named 'mmpose'`: install ViTPose/MMPose: `pip install -v -e third-party/hamer/third-party/ViTPose`
- `ModuleNotFoundError: No module named 'mmcv'`: HaMeR expects `mmcv==1.3.9` (see `third-party/hamer/setup.py`) and ViTPose/MMPose enforces `mmcv-full>=1.3.8,<=1.5.0` (see `third-party/hamer/third-party/ViTPose/mmpose/__init__.py`). In practice this usually means following HaMeR’s recommended Python/PyTorch setup.
- Missing MANO: put `MANO_RIGHT.pkl` at `third-party/hamer/_DATA/data/mano/MANO_RIGHT.pkl`

### Broken symlinks (`third-party/ViTPose`, `_DATA`)
From repo root:
```bash
rm -f third-party/ViTPose
ln -s hamer/third-party/ViTPose third-party/ViTPose
rm -f _DATA
ln -s third-party/hamer/_DATA _DATA
```

### PyTorch3D install issues
PyTorch3D wheels are specific to your Python + PyTorch + CUDA. Use https://pytorch3d.org/ to pick the correct install command.

## Acknowledgements
This implementation builds on the [MCC](https://github.com/facebookresearch/MCC) codebase, which in turn is based on [MAE](https://github.com/facebookresearch/mae).

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@article{wu2024reconstructing,
  title={Reconstructing Hand-Held Objects in 3D},
  author={Wu, Jane and Pavlakos, Georgios and Gkioxari, Georgia and Malik, Jitendra},
  journal={arXiv preprint arXiv:2404.06507,
  year={2024},
}
```
