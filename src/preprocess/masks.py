"""Foreground mask generation backends.

SAM2 modes do not use text prompts. They either use first-frame manual
point/box prompts or automatic first-frame mask candidates followed by a
selected mask ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
from PIL import Image, ImageDraw

from src.config import PreprocessConfig
from src.utils.io import read_image, save_json, write_image


def estimate_masks(
    frame_paths: List[Path],
    output_dir: Path,
    config: PreprocessConfig,
    preview_dir: Path | None = None,
    candidates_dir: Path | None = None,
    device: str = "cpu",
) -> List[Path]:
    """Estimate masks using fallback or SAM2 backends."""

    preview_dir = preview_dir or output_dir.parent / "masks_preview"
    candidates_dir = candidates_dir or output_dir.parent / "masks_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if config.mask_method in {"fallback", "saliency_fallback"}:
        return estimate_fallback_masks(frame_paths, output_dir, preview_dir)
    if config.mask_method == "sam2_manual_init":
        return SAM2MaskWorkflow(config, device=device).manual_init(frame_paths, output_dir, preview_dir)
    if config.mask_method == "sam2_auto_first_frame":
        candidates_dir.mkdir(parents=True, exist_ok=True)
        return SAM2MaskWorkflow(config, device=device).auto_first_frame(frame_paths, output_dir, preview_dir, candidates_dir)
    raise ValueError(f"Unknown mask_method: {config.mask_method}")


def estimate_fallback_masks(frame_paths: List[Path], output_dir: Path, preview_dir: Path) -> List[Path]:
    """Estimate foreground masks with the old deterministic RGB fallback."""

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    images = [read_image(p) for p in frame_paths]
    median_bg = np.median(np.stack(images, axis=0), axis=0)
    saved: List[Path] = []
    for path, image in zip(frame_paths, images):
        diff = np.linalg.norm(image - median_bg, axis=-1)
        color_saliency = np.linalg.norm(image - image.reshape(-1, 3).mean(axis=0), axis=-1)
        score = 0.6 * diff + 0.4 * color_saliency
        threshold = np.percentile(score, 65.0)
        mask = (score >= threshold).astype(np.float32)
        npy_path = output_dir / f"{path.stem}.npy"
        np.save(npy_path, mask)
        write_image(output_dir / f"{path.stem}.png", mask)
        _write_mask_preview(preview_dir / f"{path.stem}.png", image, mask)
        saved.append(npy_path)
    return saved


class SAM2MaskWorkflow:
    """SAM2 mask workflow without text prompts."""

    def __init__(self, config: PreprocessConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.repo_path = Path(config.sam2_repo)
        self.checkpoint = Path(config.sam2_checkpoint)

    def manual_init(self, frame_paths: List[Path], output_dir: Path, preview_dir: Path) -> List[Path]:
        """Initialize SAM2 on the first frame with points and/or a box, then propagate."""

        if not self.config.sam2_points and self.config.sam2_box is None:
            raise ValueError("sam2_manual_init requires preprocess.sam2_points or preprocess.sam2_box.")
        masks = self._propagate_from_prompt(frame_paths, points=self.config.sam2_points, labels=self.config.sam2_point_labels, box=self.config.sam2_box)
        save_json(
            preview_dir / "sam2_manual_prompt.json",
            {"points": self.config.sam2_points, "point_labels": self.config.sam2_point_labels, "box": self.config.sam2_box},
        )
        return _save_mask_sequence(frame_paths, masks, output_dir, preview_dir)

    def auto_first_frame(self, frame_paths: List[Path], output_dir: Path, preview_dir: Path, candidates_dir: Path) -> List[Path]:
        """Generate first-frame candidates, select one ID, then propagate it with SAM2."""

        candidates = self._generate_first_frame_candidates(frame_paths[0])
        if not candidates:
            raise RuntimeError("SAM2 automatic mask generation returned no candidates.")
        _save_candidates(frame_paths[0], candidates, candidates_dir)
        selected = self.config.selected_mask_id
        if selected < 0 or selected >= len(candidates):
            raise ValueError(f"selected_mask_id={selected} is out of range. Found {len(candidates)} candidates.")
        selected_mask = candidates[selected]["segmentation"].astype(bool)
        points, labels, box = _prompt_from_selected_mask(selected_mask)
        save_json(
            candidates_dir / "selected_mask.json",
            {"selected_mask_id": selected, "box_xyxy": box, "positive_point_xy": points[0], "num_candidates": len(candidates)},
        )
        masks = self._propagate_from_prompt(frame_paths, points=points, labels=labels, box=box)
        return _save_mask_sequence(frame_paths, masks, output_dir, preview_dir)

    def _propagate_from_prompt(
        self,
        frame_paths: List[Path],
        points: List[List[float]],
        labels: List[int],
        box: List[float] | None,
    ) -> Dict[int, np.ndarray]:
        """Run SAM2 video propagation from first-frame prompts."""

        build_sam2_video_predictor, _ = self._import_sam2()
        predictor = build_sam2_video_predictor(str(self.config.sam2_model_cfg), str(self.checkpoint), device=self.device)
        sam2_frame_dir = _prepare_sam2_jpeg_frames(frame_paths, frame_paths[0].parent.parent / "sam2_video_frames")
        inference_state = predictor.init_state(video_path=str(sam2_frame_dir))
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(inference_state)

        point_array = np.asarray(points, dtype=np.float32) if points else None
        if points and labels and len(points) != len(labels):
            raise ValueError("sam2_points and sam2_point_labels must have the same length.")
        label_array = np.asarray(labels if labels else [1] * len(points), dtype=np.int32) if points else None
        box_array = np.asarray(box, dtype=np.float32) if box is not None else None
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=self.config.sam2_obj_id,
            points=point_array,
            labels=label_array,
            box=box_array,
        )

        masks: Dict[int, np.ndarray] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if self.config.sam2_obj_id in list(out_obj_ids):
                obj_index = list(out_obj_ids).index(self.config.sam2_obj_id)
            else:
                obj_index = 0
            mask = (out_mask_logits[obj_index] > self.config.sam2_mask_threshold).detach().cpu().numpy()
            masks[int(out_frame_idx)] = np.squeeze(mask).astype(np.float32)
        return masks

    def _generate_first_frame_candidates(self, frame_path: Path) -> List[dict]:
        """Run SAM2 automatic mask generation on the first frame."""

        _, sam2_mod = self._import_sam2()
        image = (read_image(frame_path) * 255.0).astype(np.uint8)
        model = sam2_mod["build_sam2"](str(self.config.sam2_model_cfg), str(self.checkpoint), device=self.device)
        generator = sam2_mod["SAM2AutomaticMaskGenerator"](model)
        candidates = generator.generate(image)
        candidates = sorted(candidates, key=lambda m: float(m.get("area", np.asarray(m["segmentation"]).sum())), reverse=True)
        return candidates

    def _import_sam2(self):
        """Import SAM2 while supporting a local `external/sam2` checkout."""

        if self.repo_path.exists():
            sys.path.insert(0, str(self.repo_path.resolve()))
        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. Clone/install the official SAM2 repository and set "
                "preprocess.sam2_repo, preprocess.sam2_checkpoint, and preprocess.sam2_model_cfg."
            ) from exc
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")
        return build_sam2_video_predictor, {"build_sam2": build_sam2, "SAM2AutomaticMaskGenerator": SAM2AutomaticMaskGenerator}


def _save_mask_sequence(frame_paths: List[Path], masks: Dict[int, np.ndarray], output_dir: Path, preview_dir: Path) -> List[Path]:
    """Save propagated masks and previews."""

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    last_mask: np.ndarray | None = None
    for idx, frame_path in enumerate(frame_paths):
        image = read_image(frame_path)
        mask = masks.get(idx, last_mask)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        last_mask = mask
        npy_path = output_dir / f"{frame_path.stem}.npy"
        np.save(npy_path, mask)
        write_image(output_dir / f"{frame_path.stem}.png", mask)
        _write_mask_preview(preview_dir / f"{frame_path.stem}.png", image, mask)
        saved.append(npy_path)
    return saved


def _save_candidates(frame_path: Path, candidates: List[dict], candidates_dir: Path) -> None:
    """Save automatic first-frame mask candidates and an indexed preview image."""

    candidates_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(frame_path)
    height, width = image.shape[:2]
    index_image = np.zeros((height, width, 3), dtype=np.float32)
    metadata = []
    label_positions = []
    for idx, candidate in enumerate(candidates):
        mask = candidate["segmentation"].astype(np.float32)
        write_image(candidates_dir / f"mask_{idx:03d}.png", mask)
        color = _id_color(idx)
        index_image = np.where(mask[..., None] > 0.5, 0.55 * image + 0.45 * color, index_image)
        ys, xs = np.where(mask > 0.5)
        if len(xs) > 0:
            label_positions.append((idx, int(np.mean(xs)), int(np.mean(ys))))
        metadata.append(
            {
                "mask_id": idx,
                "area": float(candidate.get("area", mask.sum())),
                "bbox_xywh": [float(v) for v in candidate.get("bbox", [])],
                "predicted_iou": float(candidate.get("predicted_iou", -1.0)),
                "stability_score": float(candidate.get("stability_score", -1.0)),
            }
        )
    index_image = np.where(index_image.sum(axis=-1, keepdims=True) > 0, index_image, image * 0.35)
    labeled = Image.fromarray((np.clip(index_image, 0.0, 1.0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(labeled)
    for idx, x, y in label_positions:
        draw.rectangle((x - 6, y - 6, x + 18, y + 10), fill=(0, 0, 0))
        draw.text((x - 4, y - 6), str(idx), fill=(255, 255, 255))
    labeled.save(candidates_dir / "index_image.png")
    save_json(candidates_dir / "candidates.json", {"frame": frame_path.name, "candidates": metadata})


def _prepare_sam2_jpeg_frames(frame_paths: List[Path], output_dir: Path) -> Path:
    """Create a SAM2-compatible JPEG frame directory named 00000.jpg, ..."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_path in enumerate(frame_paths):
        target = output_dir / f"{idx:05d}.jpg"
        if target.exists():
            continue
        image = Image.open(frame_path).convert("RGB")
        image.save(target, quality=95)
    return output_dir


def _prompt_from_selected_mask(mask: np.ndarray) -> Tuple[List[List[float]], List[int], List[float]]:
    """Create a positive point and XYXY box prompt from a selected binary mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Selected SAM2 candidate mask is empty.")
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return [[cx, cy]], [1], box


def _write_mask_preview(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    """Write an image with a green mask overlay for inspection."""

    color = np.array([0.1, 0.85, 0.25], dtype=np.float32)
    overlay = np.where(mask[..., None] > 0.5, 0.55 * image + 0.45 * color, image)
    write_image(path, overlay)


def _id_color(idx: int) -> np.ndarray:
    """Deterministic RGB color for a candidate ID."""

    rng = np.random.default_rng(idx + 12345)
    return rng.uniform(0.15, 0.95, size=(3,)).astype(np.float32)
