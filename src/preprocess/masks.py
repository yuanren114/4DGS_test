"""Foreground mask generation backends.

The default production path uses Grounding DINO for first-frame text-prompt
detection, then SAM2 for mask initialization and propagation. The deterministic
RGB fallback is kept only as an explicit backup and for quick tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import shutil
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.config import PreprocessConfig
from src.utils.io import read_image, save_json, write_image


class StopAfterMaskInit(RuntimeError):
    """Raised after first-frame debug outputs are written in stop-after-init mode."""

    def __init__(self, debug_paths: Dict[str, Path]) -> None:
        self.debug_paths = debug_paths
        super().__init__("Stopped after first-frame mask initialization.")


def estimate_masks(
    frame_paths: List[Path],
    output_dir: Path,
    config: PreprocessConfig,
    preview_dir: Path | None = None,
    candidates_dir: Path | None = None,
    debug_dir: Path | None = None,
    device: str = "cpu",
) -> List[Path]:
    """Estimate masks using the configured backend."""

    preview_dir = preview_dir or output_dir.parent / "masks_preview"
    candidates_dir = candidates_dir or output_dir.parent / "masks_candidates"
    debug_dir = debug_dir or output_dir.parent / "masks_debug"

    if config.overwrite:
        for directory in [output_dir, preview_dir, candidates_dir, debug_dir, output_dir.parent / "sam2_video_frames"]:
            if directory.exists():
                shutil.rmtree(directory)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if config.mask_method in {"fallback", "saliency_fallback"}:
        return estimate_fallback_masks(frame_paths, output_dir, preview_dir)
    if config.mask_method == "gdino_sam2":
        return GDINOSAM2MaskWorkflow(config, device=device).from_text_prompt(frame_paths, output_dir, preview_dir, debug_dir)
    if config.mask_method == "sam2_manual_box":
        return GDINOSAM2MaskWorkflow(config, device=device).from_manual_box(frame_paths, output_dir, preview_dir, debug_dir)
    if config.mask_method == "sam2_manual_mask":
        return GDINOSAM2MaskWorkflow(config, device=device).from_manual_mask(frame_paths, output_dir, preview_dir, debug_dir)
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


class GDINOSAM2MaskWorkflow:
    """Grounding DINO + SAM2 mask workflow."""

    def __init__(self, config: PreprocessConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.repo_path = Path(config.sam2_repo)
        self.checkpoint = _resolve_sam2_checkpoint(config)
        self.sam2_config = config.sam2_config or config.sam2_model_cfg
        if (
            self.checkpoint.name == "sam2.1_hiera_large.pt"
            and self.sam2_config == "configs/sam2/sam2_hiera_l.yaml"
            and (self.repo_path / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml").exists()
        ):
            print("[WARN] Using local SAM2.1 checkpoint with matching SAM2.1 config.")
            self.sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    def from_text_prompt(
        self,
        frame_paths: List[Path],
        output_dir: Path,
        preview_dir: Path,
        debug_dir: Path,
    ) -> List[Path]:
        """Detect a text-prompted box on the first frame and propagate it with SAM2."""

        if not self.config.prompt:
            raise ValueError("gdino_sam2 requires preprocess.prompt or CLI --prompt.")
        box = self._detect_first_frame_box(frame_paths[0], self.config.prompt)
        _save_first_frame_box_vis(frame_paths[0], box, debug_dir / "first_frame_box.jpg")
        save_json(
            debug_dir / "first_frame_detection.json",
            {
                "prompt": self.config.prompt,
                "box_threshold": self.config.box_threshold,
                "box_xyxy": [float(v) for v in box],
            },
        )
        return self._propagate_from_box(frame_paths, box, output_dir, preview_dir, debug_dir)

    def from_manual_box(
        self,
        frame_paths: List[Path],
        output_dir: Path,
        preview_dir: Path,
        debug_dir: Path,
    ) -> List[Path]:
        """Initialize SAM2 from a user-provided first-frame XYXY box."""

        if self.config.sam2_box is None:
            raise ValueError("sam2_manual_box requires preprocess.sam2_box or CLI --sam2_box.")
        box = np.asarray(self.config.sam2_box, dtype=np.float32)
        _save_first_frame_box_vis(frame_paths[0], box, debug_dir / "first_frame_box.jpg")
        save_json(debug_dir / "manual_box.json", {"box_xyxy": [float(v) for v in box]})
        return self._propagate_from_box(frame_paths, box, output_dir, preview_dir, debug_dir)

    def from_manual_mask(
        self,
        frame_paths: List[Path],
        output_dir: Path,
        preview_dir: Path,
        debug_dir: Path,
    ) -> List[Path]:
        """Initialize SAM2 from a user-provided first-frame binary mask."""

        if not self.config.sam2_mask_path:
            raise ValueError("sam2_manual_mask requires preprocess.sam2_mask_path or CLI --sam2_mask_path.")
        first_mask = _read_binary_mask(Path(self.config.sam2_mask_path), frame_paths[0])
        _save_first_frame_box_vis(frame_paths[0], _box_from_mask(first_mask), debug_dir / "first_frame_box.jpg")
        save_json(debug_dir / "manual_mask.json", {"mask_path": self.config.sam2_mask_path})
        predictor, inference_state = self._init_sam2_state(frame_paths)
        _, _, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=self.config.sam2_obj_id,
            mask=first_mask,
        )
        first_mask = _mask_from_logits(out_mask_logits, self.config.sam2_mask_threshold)
        self._save_first_frame_debug(frame_paths[0], first_mask, debug_dir)
        if self.config.stop_after_init:
            _save_mask_sequence(frame_paths[:1], {0: first_mask}, output_dir, preview_dir)
            raise StopAfterMaskInit(_debug_paths(debug_dir))
        masks = self._propagate(predictor, inference_state)
        return _save_mask_sequence(frame_paths, masks, output_dir, preview_dir)

    def _propagate_from_box(
        self,
        frame_paths: List[Path],
        box: np.ndarray,
        output_dir: Path,
        preview_dir: Path,
        debug_dir: Path,
    ) -> List[Path]:
        """Initialize SAM2 with a first-frame box and propagate masks."""

        predictor, inference_state = self._init_sam2_state(frame_paths)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=self.config.sam2_obj_id,
            box=np.asarray(box, dtype=np.float32),
        )
        first_mask = _mask_from_logits(out_mask_logits, self.config.sam2_mask_threshold)
        self._save_first_frame_debug(frame_paths[0], first_mask, debug_dir)
        if self.config.stop_after_init:
            _save_mask_sequence(frame_paths[:1], {0: first_mask}, output_dir, preview_dir)
            raise StopAfterMaskInit(_debug_paths(debug_dir))
        masks = self._propagate(predictor, inference_state)
        return _save_mask_sequence(frame_paths, masks, output_dir, preview_dir)

    def _init_sam2_state(self, frame_paths: List[Path]):
        """Load SAM2 and prepare its JPEG video frame directory."""

        build_sam2_video_predictor = self._import_sam2()
        predictor = build_sam2_video_predictor(str(self.sam2_config), str(self.checkpoint), device=self.device)
        sam2_frame_dir = _prepare_sam2_jpeg_frames(frame_paths, frame_paths[0].parent.parent / "sam2_video_frames")
        inference_state = predictor.init_state(video_path=str(sam2_frame_dir))
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(inference_state)
        return predictor, inference_state

    def _propagate(self, predictor, inference_state) -> Dict[int, np.ndarray]:
        """Run SAM2 propagation through the video."""

        masks: Dict[int, np.ndarray] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            obj_ids = list(out_obj_ids)
            obj_index = obj_ids.index(self.config.sam2_obj_id) if self.config.sam2_obj_id in obj_ids else 0
            mask = (out_mask_logits[obj_index] > self.config.sam2_mask_threshold).detach().cpu().numpy()
            masks[int(out_frame_idx)] = np.squeeze(mask).astype(np.float32)
        return masks

    def _detect_first_frame_box(self, frame_path: Path, text_prompt: str) -> np.ndarray:
        """Run Grounding DINO on the first frame and return the best XYXY box."""

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError("transformers is required for Grounding DINO text-prompt detection.") from exc

        image_pil = Image.open(frame_path).convert("RGB")
        width, height = image_pil.size
        model_id = self.config.gdino_model_id
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        model.eval()

        inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits.sigmoid().detach().cpu()[0]
        scores = probs.max(dim=1)[0]
        keep = scores > self.config.box_threshold
        if keep.sum() == 0:
            raise RuntimeError(
                f"No Grounding DINO detection found for prompt '{text_prompt}' "
                f"with box_threshold={self.config.box_threshold}. Lower --box_threshold "
                "or use a more concrete prompt. The pipeline does not silently fall back."
            )

        best_local_idx = torch.argmax(scores[keep])
        best_global_idx = torch.where(keep)[0][best_local_idx]
        pred_boxes = outputs.pred_boxes.detach().cpu()[0]
        cx, cy, w_n, h_n = pred_boxes[best_global_idx]
        x1 = int((cx - w_n / 2) * width)
        y1 = int((cy - h_n / 2) * height)
        x2 = int((cx + w_n / 2) * width)
        y2 = int((cy + h_n / 2) * height)
        box = np.array(
            [
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)),
                max(0, min(y2, height - 1)),
            ],
            dtype=np.float32,
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise RuntimeError(f"Grounding DINO produced an invalid box: {box.tolist()}")
        return box

    def _save_first_frame_debug(self, first_frame_path: Path, first_mask: np.ndarray, debug_dir: Path) -> None:
        """Save first-frame mask checkpoints for inspection."""

        write_image(debug_dir / "first_frame_mask.png", first_mask.astype(np.float32))
        _write_mask_preview(debug_dir / "first_frame_mask_overlay.jpg", read_image(first_frame_path), first_mask)

    def _import_sam2(self):
        """Import SAM2 while supporting a local `external/sam2` checkout."""

        if self.repo_path.exists():
            sys.path.insert(0, str(self.repo_path.resolve()))
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. Clone/install the official SAM2 repository and set "
                "preprocess.sam2_repo, preprocess.sam2_config, and preprocess.sam2_ckpt."
            ) from exc
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")
        return build_sam2_video_predictor


def _resolve_sam2_checkpoint(config: PreprocessConfig) -> Path:
    """Resolve the configured SAM2 checkpoint with a local SAM2.1 compatibility fallback."""

    ckpt = Path(config.sam2_ckpt or config.sam2_checkpoint)
    if ckpt.exists():
        return ckpt
    requested_default = Path("checkpoints/sam2_hiera_large.pt")
    local_sam21 = Path("checkpoints/sam2.1_hiera_large.pt")
    if ckpt == requested_default and local_sam21.exists():
        print(f"[WARN] {requested_default} not found; using existing local checkpoint {local_sam21}.")
        return local_sam21
    return ckpt


def _save_mask_sequence(frame_paths: List[Path], masks: Dict[int, np.ndarray], output_dir: Path, preview_dir: Path) -> List[Path]:
    """Save propagated masks and previews using frame stems."""

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


def _save_first_frame_box_vis(frame_path: Path, box: np.ndarray, save_path: Path) -> None:
    """Save first frame with the detected or manual bounding box."""

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def _read_binary_mask(mask_path: Path, frame_path: Path) -> np.ndarray:
    """Read a user-provided first-frame mask and resize it to the frame if needed."""

    frame = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != frame.size:
        mask = mask.resize(frame.size, Image.NEAREST)
    return (np.asarray(mask) > 127).astype(bool)


def _box_from_mask(mask: np.ndarray) -> np.ndarray:
    """Compute an XYXY box around a non-empty binary mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("The first-frame manual mask is empty.")
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _mask_from_logits(mask_logits, threshold: float) -> np.ndarray:
    """Convert SAM2 mask logits to a float binary mask."""

    return (mask_logits[0] > threshold).detach().cpu().numpy().squeeze().astype(np.float32)


def _write_mask_preview(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    """Write an image with a green mask overlay for inspection."""

    color = np.array([0.1, 0.85, 0.25], dtype=np.float32)
    overlay = np.where(mask[..., None] > 0.5, 0.55 * image + 0.45 * color, image)
    write_image(path, overlay)


def _debug_paths(debug_dir: Path) -> Dict[str, Path]:
    """Return first-frame debug paths."""

    return {
        "first_frame_box": debug_dir / "first_frame_box.jpg",
        "first_frame_mask": debug_dir / "first_frame_mask.png",
        "first_frame_mask_overlay": debug_dir / "first_frame_mask_overlay.jpg",
    }
