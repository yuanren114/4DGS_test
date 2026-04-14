# Implementation Assumptions

This document separates the PDF-described 4DGS360 method from the engineering decisions required to make a runnable RGB-only PyTorch codebase.

## PDF Method Summary

The PDFs describe 4DGS360, a monocular dynamic object reconstruction method. The paper's method input is a sequence of RGB training frames, depth maps, and camera parameters. It uses AnchorTAP3D to create 3D point trajectories from 2D tracking anchors and a 3D tracker, initializes canonical dynamic Gaussians and hierarchical motion nodes from those trajectories, then optimizes a dynamic Gaussian representation with RGB, mask, depth, track, and ARAP-style rigidity losses. Inference renders a chosen time from training or novel camera views, including 360-degree bullet-time style views.

The available user input for this repository is stricter than the paper setup: only RGB video is assumed. Therefore depth, cameras, masks, and tracks are estimated during preprocessing. The default tracking backend is explicitly a proxy baseline, not AnchorTAP3D.

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

If depth fails, inspect normalized depth PNGs, try a larger Depth Anything model, lower image resolution only after confirming quality, or replace depth `.npy` files. If camera estimation fails, use a static-camera setting only for debugging or provide converted camera poses. If masks fail, prefer the SAM2 modes documented in README and inspect `masks_preview/` plus `masks_candidates/`. If tracks fail, inspect `tracks/tracks.npz`, then replace them with outputs from stronger tracking tools.

## SAM2 Masking Status

PDF-specified method: The PDFs use dynamic object masks but do not specify how masks are obtained from RGB-only input.

Current implementation: `mask_method: sam2_manual_init` uses first-frame points and/or a first-frame bounding box, then propagates the object mask through the video with SAM2. `mask_method: sam2_auto_first_frame` runs SAM2 automatic mask generation on the first frame, saves all candidates and an index image, selects `selected_mask_id`, converts that selected mask to a point/box prompt, and propagates it through the video. `mask_method: fallback` keeps the old RGB saliency/median-background method for quick tests.

Faithfulness: This is not directly specified by 4DGS360, but it is a stronger standard-practice engineering choice than the fallback because it avoids text prompts and produces video-propagated masks.

Why no GroundingDINO: The user requested a workflow without text prompts. SAM2 manual points/boxes and automatic first-frame masks satisfy that constraint.

Classification: MEDIUM. Mask quality affects optimization and object compactness, but it is less central than camera/depth/tracker validity.

## AnchorTAP3D Implementation Status

PDF-specified method: The PDFs specify AnchorTAP3D as a combination of BootsTAP for 2D tracking and TAPIP3D for 3D tracking. BootsTAP confidence is computed from visibility and uncertainty logits. High-confidence 2D points are lifted with depth/camera parameters and used as anchors inside a sliding-window 3D tracker. The 3D tracker is iterated within a fixed-length window, with anchors replacing reliable points after each inference iteration.

Current default implementation: `track_method: proxy_grid_lk` uses `src/preprocess/proxy_3d_tracks.py`. It samples first-frame foreground grid points, optionally applies OpenCV Lucas-Kanade optical flow, samples pseudo-depth at the tracked 2D coordinates, and unprojects with the estimated camera JSON. It saves `uv`, `xyz`, and confidence arrays to `tracks.npz`.

Current optional official-component integration: `track_method: bootstap_tapip3d_components` uses `src/preprocess/boots_tap_adapter.py` to call Google DeepMind TAP-Net's PyTorch BootsTAPIR checkpoint and `src/preprocess/tapip3d_adapter.py` to call the official TAPIP3D inference script from `zbw001/TAPIP3D`. The setup helper `scripts/setup_external_trackers.py` clones both repositories and downloads the BootsTAPIR and TAPIP3D checkpoints. This integrates the named external components as much as the public code permits, but it does not implement the paper's missing AnchorTAP3D anchor-window modification unless that code is supplied.

Faithfulness: NOT paper-faithful for the default backend. The optional official-component backend is closer, but still not a complete AnchorTAP3D implementation because the 4DGS360-specific anchor-guided TAPIP3D loop is not available in the PDFs or public code inspected here.

Why substitution exists: The PDFs describe AnchorTAP3D conceptually but do not provide executable code for the anchor-conditioned TAPIP3D modification. Public repositories provide BootsTAPIR/TAP-Net and TAPIP3D separately, so the code wraps those official components where available and keeps a proxy backend for fast runnable tests.

What is missing: Exact AnchorTAP3D window scheduling, anchor replacement logic inside TAPIP3D, confidence filtering integration during every 3D inference iteration, dynamic-mask rejection inside the tracker, and any 4DGS360-specific tracker model weights or code.

Effect on results: HIGH. Initialization quality is central to the paper's claim. Proxy tracks can overfit visible surfaces and fail precisely where AnchorTAP3D is meant to help. Official BootsTAPIR + TAPIP3D components are expected to be better, but without the anchor-guided loop they still may not reproduce paper behavior.

User must care: YES. Do not trust 360-degree reconstruction claims unless this section is resolved.

Classification: HIGH.

Environment status: Internet and GitHub access were available. The external repositories were cloned locally under `external/`, and checkpoints were downloaded under `checkpoints/`. TAP-Net BootsTAPIR checkpoint loading was verified. TAPIP3D import currently requires its compiled `pointops2` extension and related official environment setup; on this Windows Python 3.13 environment, `sophuspy` failed to build because Microsoft C++ Build Tools were missing. The limitation for running TAPIP3D here is environmental, not methodological.

## Assumption Table

### A1. RGB-only input expansion

Topic: Required modalities.

Missing or unclear: The paper method assumes RGB frames, depth maps, and camera parameters, while this repository must accept only RGB video.

Decision: Add preprocessing stages for depth, camera, masks, and 3D tracks. The default track backend is a proxy baseline; an optional official BootsTAPIR + TAPIP3D component backend is exposed.

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

### A4. 3D tracking and AnchorTAP3D gap

Topic: 2D/3D tracking.

Missing or unclear: BootsTAP and TAPIP3D are specified, but the PDFs do not provide the exact 4DGS360 AnchorTAP3D implementation that injects high-confidence 2D anchors into TAPIP3D's sliding-window inference loop.

Decision: Rename the runnable baseline to `proxy_grid_lk` and expose it through `get_3d_tracks(...)`. Add optional adapters for official BootsTAPIR and TAPIP3D components under `bootstap_tapip3d_components`.

Why reasonable: It prevents false paper-faithfulness claims while preserving a runnable baseline and a clean path to stronger external trackers.

Review needed: Yes.

Support type: Engineering workaround plus partial official-component integration.

Risk: HIGH.

### A5. Mask estimation

Topic: Dynamic foreground masks.

Missing or unclear: The PDFs use dynamic object masks but do not specify how to get them from RGB-only user video.

Decision: Use SAM2 manual initialization or SAM2 automatic first-frame mask selection for real preprocessing. Keep the old median-background/color-saliency estimator only as `fallback`.

Why reasonable: SAM2 provides promptless automatic first-frame candidates and point/box-based video propagation, so it does not require GroundingDINO or text prompts.

Review needed: Yes for real data.

Support type: Inferred from standard practice.

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

HIGH: A1 RGB-only expansion, A2 depth, A3 cameras, A4 3D tracking and AnchorTAP3D gap, A6 renderer, A7 motion simplification.

Before trusting experimental results, review all HIGH items.
