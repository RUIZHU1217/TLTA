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

## Overall architecture


### Input-Level Topology Awareness (i-LTA)

- `SuperPixelSegmentationModule` implements Algorithm 2 using the stated
  CIELAB/spatial SLIC distance in Equations (18)-(19), low-gradient center
  initialization, iterative assignment, and connectivity enforcement.
- `SuperPixelHyperGraphConstruction` implements the kernel-density and
  agglomerative mean-shift construction described by Equations (20)-(21), with
  configurable multi-scale bandwidths.
- `HyperGraphConvolutionBlock` runs first-order Chebyshev spectral convolution
  and hyperpath spatial message passing in parallel, concatenates both results,
  and applies SEB (Equations (3)-(16)).
- `DenseContextualFeatureExtraction` uses paper-specified dilation factors
  3, 6, and 12 (Equation (22)).
- `CrossAttentionCollaborativeNetwork` contains exactly five serial blocks and
  implements Equations (23)-(24).

### Feature-Level Topology Awareness (f-LTA)

- sine/cosine position encoding uses the paper-specified dimension `d=256`
  from Equation (17);
- four serial stages have the order `SwinBlock -> FA-HGC -> HGCB -> VL-FSA ->
  EL-FSA`, followed by spatial downsampling between stages;
- `FeatureAdaptiveHyperGraphConstruction` implements pooled context, dynamic
  global prototypes, multi-head similarity, and continuous incidence from
  Equations (25)-(27);
- `VertexLevelFeatureSelfAttention` and `EdgeLevelFeatureSelfAttention`
  implement Equations (28)-(31).

### Proposal-Level Topology Awareness (p-LTA)

- `ProposalPredictionNetwork` follows the Faster R-CNN RPN-like PPN described
  in Section 3.6.1: 3x3 feature reduction, regression and FC classification
  branches, proposal decoding, and NMS;
- proposal/GT IoU `>=0.5` is positive and `<0.5` is negative, as stated in
  Algorithm 1;
- `ProposalGuidedHyperGraphConstruction` contains distinct `PP-HGC` and
  `NP-HGC` classes and implements cosine similarity, top-k hyperedges, and
  Equation (34) edge weights;
- the two hypergraphs are processed by parallel HGCB paths, concatenated,
  flattened to a fixed-length feature sequence, and passed to the encoder;
- `ContrastiveDeNoisingFeatureDecoding` uses the exact manuscript thresholds
  `lambda_1=0.1`, `lambda_2=0.7`, positive-gradient IoU threshold `0.3`, and
  negative threshold `sigma=0.3`;
- `CDNFLoss` uses `lambda_reg=1.0`, `lambda_cls=2.0`, and `lambda_neg=0.5`
  from Equation (38).

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




