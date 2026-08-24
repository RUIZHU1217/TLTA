# TLTA

 **"Triple-Level Topology Awareness Using Hypergraph for Marine Ship Surveillance from SAR Imagery"**.





## Introduction

TLTA models high-order relationships at three serial levels:

1. input-level topology awareness (i-LTA) groups pixels into super-pixels,
   constructs a super-pixel hypergraph, and combines topology features with
   dense pixel context;
2. feature-level topology awareness (f-LTA) constructs continuous,
   feature-adaptive hypergraphs in a four-stage backbone and applies
   vertex-/edge-level attention;
3. proposal-level topology awareness (p-LTA) builds separate positive and
   negative proposal hypergraphs before transformer encoding and contrastive
   denoising feature decoding.


## Installation

Python 3.10 or newer is recommended.

```bash
cd TLTA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the repository checks after installation:

```bash
python tools/validate_repo.py
python tools/smoke_test.py
pytest -q
```

## Dataset preparation

The manuscript evaluates SSDD and HRSID. Obtain them from the dataset locations
cited by the paper:

- SSDD: <https://github.com/TianwenZhang0825/Official-SSDD>
- HRSID: <https://github.com/chaozhong2010/HRSID>

Expected directory structure:

```text
data/
├── SSDD/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── annotations/
│       ├── instances_train.json
│       └── instances_test.json
└── HRSID/
    ├── images/
    │   ├── train/
    │   └── test/
    └── annotations/
        ├── instances_train.json
        └── instances_test.json
```

The manuscript states 800x800 input and batch size 4 for SSDD, and 1024x1024
input and batch size 2 for HRSID. The resize policy itself is not described;
this repository uses direct resize and marks it as an implementation detail.

## Training

Both dataset configurations reproduce the explicitly reported optimization
settings: SGD for 320 epochs, initial learning rate 0.04, 10x reductions at
epochs 280 and 300, Nesterov momentum 0.99, and weight decay 0.005.

```bash
python train.py --config configs/ssdd.yaml --device cuda
python train.py --config configs/hrsid.yaml --device cuda
```

Resume training with:

```bash
python train.py \
  --config configs/ssdd.yaml \
  --resume outputs/ssdd/last.pth \
  --device cuda
```

Use `--seed` to override the reproducible seed in the YAML. Training writes
checkpoints and JSONL metrics under the configured output directory.

## Evaluation

```bash
python test.py \
  --config configs/ssdd.yaml \
  --checkpoint outputs/ssdd/last.pth \
  --device cuda
```

`pycocotools` reports the COCO bounding-box metrics used by the paper: AP,
AP50, AP75, AP_S, AP_M, and AP_L. Predictions are rescaled to the original
image size before evaluation.

## Inference

```bash
python demo.py \
  --config configs/ssdd.yaml \
  --checkpoint outputs/ssdd/last.pth \
  --image path/to/sar_image.jpg \
  --output demo_output.png \
  --device cuda
```

The score threshold can be overridden with `--score-threshold`.

## Repository structure

```text
TLTA/
├── README.md
├── PAPER_TO_CODE.md
├── requirements.txt
├── train.py
├── test.py
├── demo.py
├── configs/
│   ├── ssdd.yaml
│   ├── hrsid.yaml
│   └── smoke.yaml
├── datasets/
│   ├── build.py
│   ├── coco.py
│   └── transforms.py
├── models/
│   ├── tlta.py
│   ├── hypergraph.py
│   ├── i_lta.py
│   ├── cacn.py
│   ├── f_lta.py
│   ├── p_lta.py
│   └── cdn_fd.py
├── utils/
│   ├── boxes.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── evaluator.py
│   ├── seed.py
│   └── visualization.py
├── tools/
│   ├── smoke_test.py
│   ├── validate_repo.py
│   └── voc_to_coco.py
├── tests/
│   └── test_modules.py
└── assets/
```




