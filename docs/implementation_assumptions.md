# Implementation Assumptions

This document separates the PDF-described 4DGS360 method from the engineering decisions required to make a runnable RGB-only PyTorch codebase.

## PDF Method Summary

The PDFs describe 4DGS360, a monocular dynamic object reconstruction method. The paper's method input is a sequence of RGB training frames, depth maps, and camera parameters. It uses AnchorTAP3D to create 3D point trajectories from 2D tracking anchors and a 3D tracker, initializes canonical dynamic Gaussians and hierarchical motion nodes from those trajectories, then optimizes a dynamic Gaussian representation with RGB, mask, depth, track, and ARAP-style rigidity losses. Inference renders a chosen time from training or novel camera views, including 360-degree bullet-time style views.

The available user input for this repository is stricter than the paper setup: only RGB video is assumed. Therefore depth, cameras, masks, and tracks are estimated or approximated during preprocessing.

## Camera And Depth Decision Report

### Depth Method

Selected method: Depth Anything v2 Small via Hugging Face `transformers` when available, wrapped by `src/preprocess/depth.py`. If model loading fails, the code uses a deterministic RGB saliency/luminance fallback so quick tests and offline wiring checks remain runnable.

Why: Depth Anything v2 is a strong practical monocular depth estimator, PyTorch-accessible, reproducible, and appropriate for RGB-only videos. The fallback is not paper-faithful and is only for smoke testing.

Risks: Monocular depth is scale-ambiguous, may be temporally inconsistent, and can fail on transparent, reflective, low-texture, or unusual objects. Depth errors directly affect unprojection, tracks, Gaussian initialization, and depth loss.

Manual intervention: Review depth previews under `outputs/run_*/preprocess/depth`. If they are wrong, use a better depth model or externally generated depth converted to the saved `.npy` structure.

Classification: HIGH. This is an engineering workaround because only RGB video is available.

### Camera Estimation Method

Selected method: `colmap_or_simple_vo`. The code detects COLMAP availability but writes a simple object-centric proxy camera trajectory by default. This keeps the pipeline runnable from RGB-only input. The camera schema is documented in `outputs/run_*/preprocess/camera/cameras.json`.

Why: The PDFs accept dataset sensors or pretrained estimators for camera parameters, but the user has no camera intrinsics/extrinsics. COLMAP is a standard practical option for static or moderately dynamic scenes, but it is external and may fail on dynamic object videos. The proxy trajectory allows quick testing and end-to-end inspection without pretending the estimates are metrically correct.

Risks: Proxy cameras are not metrically reliable. Dynamic scenes violate COLMAP assumptions. Wrong camera poses make 3D reconstruction and novel-view rendering unreliable.

Manual intervention: For real experiments, run COLMAP, DROID-SLAM, DUSt3R/MASt3R, or another robust RGB camera estimator externally, then convert its intrinsics/extrinsics into the repository camera JSON schema.

Classification: HIGH. This affects method validity and paper faithfulness.

### Failure Advice

If depth fails, inspect normalized depth PNGs, try a larger Depth Anything model, lower image resolution only after confirming quality, or replace depth `.npy` files. If camera estimation fails, use a static-camera setting only for debugging or provide converted camera poses. If masks or tracks fail, inspect `masks/*.png` and `tracks/tracks.npz`, then replace them with outputs from SAM/Track-Anything/CoTracker/BootsTAP style tools.

## Assumption Table

### A1. RGB-only input expansion

Topic: Required modalities.

Missing or unclear: The paper method assumes RGB frames, depth maps, and camera parameters, while this repository must accept only RGB video.

Decision: Add preprocessing stages for depth, camera, masks, and pseudo-tracks.

Why reasonable: This is the only way to run the pipeline from the stated input.

Review needed: Yes.

Support type: Engineering workaround because inputs are limited to RGB video.

Risk: HIGH.

### A2. Depth Anything v2 as pseudo-depth

Topic: Depth estimation.

Missing or unclear: The PDFs cite dataset sensor depth or pretrained estimators but do not prescribe a concrete RGB-only depth implementation.

Decision: Use Depth Anything v2 Small when available; fallback to deterministic pseudo-depth only for quick tests.

Why reasonable: Depth Anything v2 is strong, public, and PyTorch-accessible.

Review needed: Yes.

Support type: Inferred from standard practice plus RGB-only workaround.

Risk: HIGH.

### A3. Camera proxy trajectory

Topic: Intrinsics and extrinsics.

Missing or unclear: No ground-truth camera parameters are available. The PDFs do not define a complete camera recovery pipeline.

Decision: Save pinhole intrinsics from assumed 60-degree FOV and simple object-centric proxy extrinsics unless the user replaces them.

Why reasonable: It keeps code runnable and exposes the exact camera schema.

Review needed: Yes, before trusting results.

Support type: Engineering workaround because only RGB video is available.

Risk: HIGH.

### A4. AnchorTAP3D substitution

Topic: 2D/3D tracking.

Missing or unclear: BootsTAP and TAPIP3D are described but not implemented in enough detail to reproduce inside this repository.

Decision: Implement a grid plus optical-flow pseudo-track builder and unproject tracks using pseudo-depth/cameras. Mark it as an AnchorTAP3D substitute.

Why reasonable: It provides reusable track tensors and makes the model/training path runnable.

Review needed: Yes.

Support type: Engineering workaround.

Risk: HIGH.

### A5. Mask estimation

Topic: Dynamic foreground masks.

Missing or unclear: The PDFs use dynamic object masks but do not specify how to get them from RGB-only user video.

Decision: Use a median-background/color-saliency fallback mask estimator.

Why reasonable: It is deterministic and inspectable; stronger SAM/video segmentation can replace the saved masks.

Review needed: Yes for real data.

Support type: Engineering workaround.

Risk: MEDIUM.

### A6. Renderer implementation

Topic: Gaussian rasterization.

Missing or unclear: The PDFs rely on differentiable 3DGS-style rasterization but do not provide code.

Decision: Implement a pure PyTorch isotropic point-Gaussian splatter rather than a CUDA 3DGS rasterizer.

Why reasonable: It is portable and runnable, but slower and less faithful.

Review needed: Yes if comparing experiments to the paper.

Support type: Engineering workaround.

Risk: HIGH.

### A7. Motion representation simplification

Topic: Hierarchical motion.

Missing or unclear: HiMoR-style hierarchy and motion bases are described conceptually, with incomplete implementation details.

Decision: Use canonical Gaussians with K-nearest node interpolation and per-frame node translations.

Why reasonable: It preserves the main structural idea while staying compact and trainable.

Review needed: Yes.

Support type: Inferred from standard practice and simplified engineering implementation.

Risk: HIGH.

### A8. Loss weights

Topic: Optimization.

Missing or unclear: Supplement gives several weights, but some learning rates are PDF-extraction corrupted and LPIPS/D-SSIM exact implementation is not code-specified.

Decision: Use the reported weights where practical; set LPIPS default weight to 0 for dependency-light quick tests; implement a small SSIM-style D-SSIM substitute.

Why reasonable: Keeps the loss terms inspectable and runnable.

Review needed: Medium priority.

Support type: Partly directly supported by PDFs, partly engineering detail.

Risk: MEDIUM.

### A9. Densification

Topic: Adaptive density control.

Missing or unclear: The supplement mentions 3DGS adaptive density control but not exact schedule.

Decision: Do not implement densification in the baseline; expose `num_gaussians` as a config value.

Why reasonable: Densification is complex and not necessary for structural runnable code.

Review needed: Yes for quality-oriented experiments.

Support type: Engineering workaround.

Risk: MEDIUM.

### A10. Evaluation

Topic: Metrics.

Missing or unclear: RGB-only user data lacks synchronized test cameras and ground truth masks.

Decision: Provide visualization outputs and training losses, not benchmark metrics.

Why reasonable: The required ground truth does not exist in the stated input.

Review needed: Low unless benchmarking.

Support type: Engineering workaround.

Risk: LOW.

## Assumptions By Risk

LOW: A10 evaluation output scope.

MEDIUM: A5 masks, A8 loss defaults, A9 densification omitted.

HIGH: A1 RGB-only expansion, A2 depth, A3 cameras, A4 AnchorTAP3D substitution, A6 renderer, A7 motion simplification.

Before trusting experimental results, review all HIGH items.
