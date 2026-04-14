# 4DGS360 RGB-Only PyTorch Baseline

This repository implements a complete, runnable, modular PyTorch baseline inspired by the two PDFs in `docs/`. The target method is 4DGS360: 360-degree dynamic object reconstruction from monocular video using 3D track initialization, canonical dynamic Gaussians, node-guided motion, differentiable rendering, and RGB/depth/mask/track/ARAP losses.

The PDFs assume signals that are not available here. This code starts from RGB video only and builds practical preprocessing outputs for depth, cameras, masks, and tracks. Those preprocessing stages are explicitly documented as engineering substitutions in `docs/implementation_assumptions.md`.

## Paper Mapping

Implemented directly from the PDFs at a structural level:

- RGB frame sequence preprocessing and reuse.
- Pseudo-depth and camera inputs saved for training.
- 3D trajectory-based Gaussian initialization.
- Canonical Gaussian representation with time-dependent deformation.
- Node-guided motion interpolation.
- RGB, mask, depth, tracking, and ARAP-style training losses.
- Training-view and bullet-time style inference outputs.

Engineering substitutions:

- Depth is estimated from RGB with Depth Anything v2 when available, otherwise a fallback for quick tests.
- Camera parameters are saved in a reproducible JSON schema, with a simple proxy trajectory by default.
- The default tracking backend is `proxy_grid_lk`: foreground grid tracks, optional OpenCV optical flow, and depth/camera unprojection. This is not AnchorTAP3D.
- Optional official-component adapters are provided for Google DeepMind BootsTAPIR and the official TAPIP3D repository, but the full 4DGS360 AnchorTAP3D anchor-window algorithm remains unavailable unless supplied separately.
- Rendering uses a pure PyTorch isotropic Gaussian splatter rather than the production CUDA 3DGS rasterizer.

## Component Faithfulness

| Component | PDF Method | Current Implementation | Faithfulness | Importance | Needs Review |
| --------- | ---------- | ---------------------- | ------------ | ---------- | ------------ |
| Depth | Dataset/sensor or pretrained depth input | Depth Anything v2 wrapper, fallback pseudo-depth for quick tests | MEDIUM | HIGH | YES |
| Camera | Dataset/sensor or pretrained camera parameters | COLMAP-ready interface, simple proxy trajectory by default | LOW | HIGH | YES |
| AnchorTAP3D | BootsTAP + TAPIP3D with anchor-guided sliding-window 3D tracking | Default `proxy_grid_lk`; optional `bootstap_tapip3d_components` adapters without the missing anchor-window modification | LOW | HIGH | YES |
| Masks | Dynamic object masks | RGB saliency/median fallback masks | LOW | MEDIUM | YES |
| Dynamic Gaussians | Canonical 3DGS with hierarchical motion | Compact PyTorch Gaussians with KNN node translations | MEDIUM-LOW | HIGH | YES |
| Renderer | Differentiable 3DGS rasterizer | Pure PyTorch isotropic splatter | LOW | HIGH | YES |
| Losses | RGB, mask, depth, track, ARAP | Implemented approximations with logged terms | MEDIUM | MEDIUM | YES |

## Installation

Create an environment with Python 3.10+ and install:

```bash
pip install -r requirements.txt
```

For real videos, `opencv-python` is needed for frame extraction and optical flow. `transformers` enables Depth Anything v2 loading. If Depth Anything weights cannot be downloaded, the code still runs with the documented fallback, but quality should not be trusted.

To set up the optional official tracker components:

```bash
python scripts/setup_external_trackers.py
pip install -e external/tapnet
```

This clones:

- `https://github.com/google-deepmind/tapnet.git`
- `https://github.com/zbw001/TAPIP3D.git`

and downloads:

- `checkpoints/bootstapir_checkpoint_v2.pt`
- `checkpoints/tapip3d_final.pth`

TAPIP3D's official inference additionally requires its compiled `pointops2` extension and the environment described in `external/TAPIP3D/README.md`. On Windows/Python 3.13 this may require Microsoft C++ Build Tools or, more practically, a separate Python 3.10 CUDA environment.

## Quick Test

Run the full pipeline on a tiny synthetic RGB sequence:

```bash
python scripts/quick_test.py
```

This creates:

```text
outputs/run_quick_test/
  config.yaml
  logs.txt
  preprocess/
  checkpoints/final.pt
  debug/
  visualizations/
  inference/
```

The quick test prints tensor shapes, loss terms, preprocessing paths, and saves debug images. It is for wiring validation, not reconstruction quality.

## Preprocess A Video

```bash
python scripts/preprocess_video.py --input_video path/to/video.mp4 --max_frames 80
```

You can also pass a directory of RGB frames:

```bash
python scripts/preprocess_video.py --input_video path/to/frames
```

Preprocessing writes:

```text
outputs/run_YYYYMMDD_HHMM/preprocess/
  frames/
  depth/
  camera/cameras.json
  masks/
  tracks/tracks.npz
  metadata/preprocess_manifest.json
```

The default tracker backend is `proxy_grid_lk`. To request the optional official-component backend after setting up the external repositories and TAPIP3D environment, set this in a config file:

```yaml
preprocess:
  track_method: bootstap_tapip3d_components
  bootstap_repo: external/tapnet
  bootstap_checkpoint: checkpoints/bootstapir_checkpoint_v2.pt
  tapip3d_repo: external/TAPIP3D
  tapip3d_checkpoint: checkpoints/tapip3d_final.pth
```

This backend is closer to the PDFs because it uses BootsTAPIR and TAPIP3D components, but it is still not a complete paper-faithful AnchorTAP3D implementation without the missing anchor-guided TAPIP3D loop.

## Train

Train from a new video:

```bash
python scripts/train.py --input_video path/to/video.mp4 --iterations 1000
```

Train from an existing preprocessed run:

```bash
python scripts/train.py --run_dir outputs/run_YYYYMMDD_HHMM --iterations 1000
```

Check `logs.txt` for loss values and tensor shapes. Check `debug/` and `visualizations/` for saved renders.

## Inference

```bash
python scripts/inference.py --run_dir outputs/run_YYYYMMDD_HHMM
```

This writes training-view and simple bullet-time images to:

```text
outputs/run_YYYYMMDD_HHMM/inference/
```

## Configuration

The default config lives in `src/config.py` and is saved to each run as `config.yaml`. Important settings include:

- `preprocess.input_video`
- `preprocess.frame_stride`
- `preprocess.max_frames`
- `preprocess.depth_method`
- `preprocess.camera_method`
- `preprocess.track_method`
- `model.num_gaussians`
- `model.num_nodes`
- `train.iterations`
- `train.lr`
- `train.device`
- `inference.num_bullet_time_views`

You can pass a YAML file with `--config`.

## Known Limitations

This is runnable and inspectable, but it is not a full paper-faithful reproduction. The most important gaps are camera estimation, Depth Anything pseudo-depth quality, missing full AnchorTAP3D anchor-guided 3D tracking, simplified node motion, and pure PyTorch splatting instead of CUDA 3DGS rasterization.

Before trusting results on your own data, review the HIGH-risk assumptions in `docs/implementation_assumptions.md`, especially camera poses, pseudo-depth, and track backend. For experimental use, replace the proxy camera JSON with COLMAP, DROID-SLAM, DUSt3R/MASt3R, or another robust estimator, and use the official tracker adapters or a true 4DGS360 AnchorTAP3D implementation if available.
