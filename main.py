import os
import json
import signal
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence, Set, Any
from multiprocessing import Pool, cpu_count
from functools import partial

import fitz  # PyMuPDF (Still needed for drawing output PDF)
import pypdfium2 as pdfium
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
import numpy as np

try:
    import pymupdf4llm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pymupdf4llm = None  # type: ignore

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model options
MODEL_SIZE = 1024
REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
WEIGHTS_FILE = f"doclayout_yolo_docstructbench_imgsz{MODEL_SIZE}.pt"

# Detection settings
CONF_THRESHOLD = 0.25

# Multiprocessing settings
NUM_WORKERS = None  # None = auto (cpu_count - 1), or set to specific number like 4
USE_MULTIPROCESSING = True  # Set to False to disable parallel processing entirely

# ----------------------------------------------------------------------
# Color map for the layout classes
# ----------------------------------------------------------------------
CLASS_COLORS = {
    "text": (0, 128, 0),          # Dark Green
    "title": (192, 0, 0),        # Dark Red
    "figure": (0, 0, 192),       # Dark Blue
    "table": (218, 165, 32),     # Goldenrod (Dark Yellow)
    "list": (128, 0, 128),       # Purple
    "header": (0, 128, 128),     # Teal
    "footer": (100, 100, 100),   # Dark Gray
    "figure_caption": (0, 0, 128), # Navy
    "table_caption": (139, 69, 19),  # Saddle Brown
    "table_footnote": (128, 0, 128), # Purple
}

# Global model instance (will be None in worker processes until loaded)
_model = None
_shutdown_requested = False

# ----------------------------------------------------------------------
# Signal handler for graceful shutdown
# ----------------------------------------------------------------------
def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        logger.warning("\n⚠️  Interrupt received! Finishing current page and shutting down gracefully...")
        logger.warning("Press Ctrl+C again to force quit (may leave incomplete files)")
    else:
        logger.error("\n❌ Force quit requested. Exiting immediately.")
        sys.exit(1)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ----------------------------------------------------------------------
# Model loader function
# ----------------------------------------------------------------------
def get_model():
    """Lazy load the model (only once per process)."""
    global _model
    if _model is None:
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILE)
        _model = YOLOv10(weights_path)
        logger.info(f"✓ Model loaded in worker process (PID: {os.getpid()})")
    return _model

# ----------------------------------------------------------------------
# Worker initialization function
# ----------------------------------------------------------------------
def init_worker():
    """Initialize worker process - loads model once at startup."""
    try:
        get_model()
        logger.success(f"Worker {os.getpid()} ready")
    except Exception as e:
        logger.error(f"Failed to initialize worker {os.getpid()}: {e}")
        raise

# ----------------------------------------------------------------------
# Run layout detection on a single page image (YOLO)
# ----------------------------------------------------------------------
def detect_page(pil_img: Image.Image) -> List[dict]:
    """Detect layout elements using YOLO model."""
    model = get_model()  # Will return already-loaded model in worker
    img_cv = np.array(pil_img)
    results = model.predict(
        img_cv,
        imgsz=MODEL_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )
    dets = []
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls.item())
        name = results[0].names[cls_id]
        conf = float(box.conf.item())
        x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().tolist()
        dets.append({
            "name": name,
            "bbox": [x0, y0, x1, y1],
            "conf": conf,
            "source": "yolo",
            "index": i
        })
    return dets

# ----------------------------------------------------------------------
# Crop & save figure/table regions (with captions)
# ----------------------------------------------------------------------
def get_union_box(box1: List[float], box2: List[float]) -> List[float]:
    """Get the bounding box enclosing two boxes."""
    x0 = min(box1[0], box2[0])
    y0 = min(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    y1 = max(box1[3], box2[3])
    return [x0, y0, x1, y1]

def collect_caption_elements(
    element: Dict,
    all_dets: List[Dict],
    target_name: str,
    max_vertical_gap: float = 60.0,
    min_overlap: float = 0.25,
) -> List[Dict]:
    """
    Collect contiguous caption detections directly below a figure/table.
    """
    base_box = element["bbox"]
    base_bottom = base_box[3]
    selected: List[Dict] = []
    last_bottom = base_bottom

    relevant = [
        d for d in all_dets
        if d["name"] == target_name and d["bbox"][1] >= base_bottom - 5
    ]

    relevant.sort(key=lambda d: d["bbox"][1])

    for cand in relevant:
        cand_box = cand["bbox"]
        top = cand_box[1]
        if selected and top - last_bottom > max_vertical_gap:
            break

        if selected:
            overlap = _horizontal_overlap_ratio(selected[-1]["bbox"], cand_box)
        else:
            overlap = _horizontal_overlap_ratio(base_box, cand_box)

        if overlap < min_overlap:
            continue

        selected.append(cand)
        last_bottom = cand_box[3]

    return selected


def collect_title_and_text_segments(
    element: Dict,
    all_dets: List[Dict],
    processed_indices: Set[int],
    settings: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Locate a title below the element and any contiguous text blocks directly beneath it.
    """
    if settings is None:
        settings = TITLE_TEXT_ASSOCIATION

    if not element.get("bbox"):
        return [], []

    figure_box = element["bbox"]
    figure_bottom = figure_box[3]

    candidates = [
        d for d in all_dets
        if d.get("bbox") and d["index"] not in processed_indices
    ]
    candidates.sort(key=lambda d: d["bbox"][1])

    titles: List[Dict] = []
    texts: List[Dict] = []

    for idx, det in enumerate(candidates):
        if det["name"] != "title":
            continue

        title_box = det["bbox"]
        if title_box[1] < figure_bottom - 5:
            continue

        vertical_gap = title_box[1] - figure_bottom
        if vertical_gap > settings["max_title_gap"]:
            break

        overlap = _horizontal_overlap_ratio(figure_box, title_box)
        if overlap < settings["min_overlap"]:
            continue

        titles.append(det)
        last_bottom = title_box[3]

        for follower in candidates[idx + 1 :]:
            if follower["name"] == "title":
                break
            if follower["name"] != "text":
                continue
            text_box = follower["bbox"]
            if text_box[1] < title_box[1]:
                continue

            gap = text_box[1] - last_bottom
            if gap > settings["max_text_gap"]:
                break

            if _horizontal_overlap_ratio(title_box, text_box) < settings["min_overlap"]:
                continue

            texts.append(follower)
            last_bottom = text_box[3]

        break

    return titles, texts


def save_layout_elements(pil_img: Image.Image, page_num: int, 
                         dets: List[dict], out_dir: Path) -> List[dict]:
    """Save figure and table crops, merging captions."""
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    infos = []
    fig_count = 0
    tab_count = 0
    
    processed_indices = set()

    for i, d in enumerate(dets):
        if d["index"] in processed_indices:
            continue
        
        name = d["name"].lower()
        final_box = d["bbox"]
        caption_segments: List[Dict] = []
        title_segments: List[Dict] = []
        text_segments: List[Dict] = []
        
        if name == "figure":
            elem_type = "figure"
            path_template = fig_dir / f"page_{page_num + 1}_fig_{fig_count}.png"
            fig_count += 1
            caption_segments = collect_caption_elements(d, dets, "figure_caption")
            for cap in caption_segments:
                final_box = get_union_box(final_box, cap["bbox"])
                processed_indices.add(cap["index"])
            title_segments, text_segments = collect_title_and_text_segments(
                d, dets, processed_indices
            )
            for seg in title_segments + text_segments:
                final_box = get_union_box(final_box, seg["bbox"])
                processed_indices.add(seg["index"])
        
        elif name == "table":
            elem_type = "table"
            path_template = tab_dir / f"page_{page_num + 1}_tab_{tab_count}.png"
            tab_count += 1
            caption_segments = collect_caption_elements(d, dets, "table_caption")
            for cap in caption_segments:
                final_box = get_union_box(final_box, cap["bbox"])
                processed_indices.add(cap["index"])
        else:
            continue
            
        x0, y0, x1, y1 = map(int, final_box)
        crop = pil_img.crop((x0, y0, x1, y1))
        
        if crop.mode == "CMYK":
            crop = crop.convert("RGB")
            
        crop.save(path_template)
        
        info_data = {
            "type": elem_type,
            "page": page_num + 1,
            "bbox_pixels": final_box,
            "conf": d["conf"],
            "source": d.get("source", "yolo"),
            "image_path": str(path_template.relative_to(out_dir)),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
            "page_width": pil_img.width,
            "page_height": pil_img.height,
        }
        if caption_segments:
            info_data["captions"] = [
                {
                    "bbox": cap["bbox"],
                    "conf": cap.get("conf"),
                    "index": cap["index"],
                    "source": cap.get("source"),
                    "page": page_num + 1,
                }
                for cap in caption_segments
            ]
        if title_segments:
            info_data["titles"] = [
                {
                    "bbox": seg["bbox"],
                    "conf": seg.get("conf"),
                    "index": seg["index"],
                    "source": seg.get("source"),
                    "page": page_num + 1,
                }
                for seg in title_segments
            ]
        if text_segments:
            info_data["texts"] = [
                {
                    "bbox": seg["bbox"],
                    "conf": seg.get("conf"),
                    "index": seg["index"],
                    "source": seg.get("source"),
                    "page": page_num + 1,
                }
                for seg in text_segments
            ]
        
        infos.append(info_data)
    
    return infos


TABLE_STITCH_TOLERANCES = {
    "x_tol": 60,
    "y_tol": 60,
    "width_tol": 120,
    "height_tol": 120,
}

CROSS_PAGE_CAPTION_THRESHOLDS = {
    "max_top_ratio": 0.35,
    "max_top_pixels": 220,
    "x_tol": 120,
    "width_tol": 200,
    "min_overlap": 0.05,
}

TITLE_TEXT_ASSOCIATION = {
    "max_title_gap": 220,
    "max_text_gap": 160,
    "min_overlap": 0.2,
}


def _horizontal_overlap_ratio(box1: List[float], box2: List[float]) -> float:
    """Compute horizontal overlap ratio between two bounding boxes."""
    x_left = max(box1[0], box2[0])
    x_right = min(box1[2], box2[2])
    overlap = max(0.0, x_right - x_left)
    if overlap <= 0:
        return 0.0
    width_union = max(box1[2], box2[2]) - min(box1[0], box2[0])
    if width_union <= 0:
        return 0.0
    return overlap / width_union


def _bbox_to_rect(bbox: List[float]) -> Tuple[int, int, int, int]:
    """Convert [x0, y0, x1, y1] into (x, y, w, h)."""
    x0, y0, x1, y1 = bbox
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _open_table_image(elem: Dict, out_dir: Path) -> Optional[Image.Image]:
    """Open a table image relative to the output directory."""
    image_path = out_dir / elem["image_path"]
    if not image_path.exists():
        logger.warning(f"Missing table crop for stitching: {image_path}")
        return None
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _pad_width(img: Image.Image, target_width: int) -> Image.Image:
    if img.width >= target_width:
        return img
    canvas = Image.new("RGB", (target_width, img.height), color=(255, 255, 255))
    canvas.paste(img, (0, 0))
    return canvas


def _pad_height(img: Image.Image, target_height: int) -> Image.Image:
    if img.height >= target_height:
        return img
    canvas = Image.new("RGB", (img.width, target_height), color=(255, 255, 255))
    canvas.paste(img, (0, 0))
    return canvas


def _append_segment_image(
    base_img: Image.Image,
    segment_img: Image.Image,
    resize_to_base: bool = False,
) -> Image.Image:
    """Append segment image below base image with optional width alignment."""
    if base_img.mode != "RGB":
        base_img = base_img.convert("RGB")
    if segment_img.mode != "RGB":
        segment_img = segment_img.convert("RGB")

    if resize_to_base and segment_img.width > 0 and base_img.width > 0:
        segment_img = segment_img.resize(
            (
                base_img.width,
                max(1, int(segment_img.height * (base_img.width / segment_img.width))),
            ),
            Image.Resampling.LANCZOS,
        )

    target_width = max(base_img.width, segment_img.width)
    base_img = _pad_width(base_img, target_width)
    segment_img = _pad_width(segment_img, target_width)

    stitched = Image.new(
        "RGB",
        (target_width, base_img.height + segment_img.height),
        color=(255, 255, 255),
    )
    stitched.paste(base_img, (0, 0))
    stitched.paste(segment_img, (0, base_img.height))
    return stitched


def _render_pdf_page(
    pdf_doc: pdfium.PdfDocument,
    page_index: int,
    scale: float,
    cache: Dict[int, Image.Image],
) -> Optional[Image.Image]:
    """Render a PDF page to a PIL image with caching."""
    if page_index in cache:
        return cache[page_index]

    try:
        page = pdf_doc[page_index]
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()
        page.close()
    except Exception as exc:
        logger.error(f"Failed to render page {page_index + 1} for caption stitching: {exc}")
        return None

    cache[page_index] = pil_img
    return pil_img


def _crop_pdf_region(
    page_img: Optional[Image.Image], bbox: List[float]
) -> Optional[Image.Image]:
    """Crop a region from a rendered PDF page."""
    if page_img is None:
        return None

    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page_img.width, max(x0 + 1, x1))
    y1 = min(page_img.height, max(y0 + 1, y1))

    if x0 >= x1 or y0 >= y1:
        return None

    crop = page_img.crop((x0, y0, x1, y1))
    if crop.mode == "CMYK":
        crop = crop.convert("RGB")
    return crop


def write_markdown_document(pdf_path: Path, out_dir: Path) -> Optional[Path]:
    """
    Extract markdown text from a PDF using PyMuPDF4LLM and write it to disk.
    """
    if pymupdf4llm is None:
        logger.warning(
            "Skipping markdown extraction for %s because pymupdf4llm is not installed.",
            pdf_path.name,
        )
        return None

    try:
        markdown_content = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:
        logger.error(f"  Failed to create markdown for {pdf_path.name}: {exc}")
        return None

    if isinstance(markdown_content, list):
        markdown_content = "\n\n".join(
            part for part in markdown_content if isinstance(part, str)
        )

    if not isinstance(markdown_content, str):
        logger.error(
            f"  Unexpected markdown output type {type(markdown_content)} for {pdf_path.name}"
        )
        return None

    markdown_content = markdown_content.strip()
    if not markdown_content:
        logger.warning(f"  No textual content extracted from {pdf_path.name}")
        return None

    if not markdown_content.endswith("\n"):
        markdown_content += "\n"

    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(markdown_content, encoding="utf-8")
    logger.info(f"  Saved markdown to {md_path.name}")
    return md_path


def _collect_text_under_title_cross_page(
    title_det: Dict,
    sorted_dets: List[Dict],
    start_idx: int,
    page_idx: int,
    used_indices: Set[Tuple[int, int]],
    settings: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Collect text elements directly below a title on the next page."""
    if settings is None:
        settings = TITLE_TEXT_ASSOCIATION
    texts: List[Dict] = []
    title_box = title_det["bbox"]
    last_bottom = title_box[3]

    for follower in sorted_dets[start_idx + 1 :]:
        det_index = follower.get("index")
        if det_index is None or (page_idx, det_index) in used_indices:
            continue

        if follower["name"] == "title":
            break

        if follower["name"] != "text":
            continue

        text_box = follower["bbox"]
        if text_box[1] < title_box[1]:
            continue

        gap = text_box[1] - last_bottom
        if gap > settings["max_text_gap"]:
            break

        if _horizontal_overlap_ratio(title_box, text_box) < settings["min_overlap"]:
            continue

        texts.append(follower)
        last_bottom = text_box[3]

    return texts


def attach_cross_page_figure_captions(
    elements: List[Dict],
    all_dets: Sequence[Optional[List[Dict[str, Any]]]],
    pdf_bytes: bytes,
    out_dir: Path,
    scale: float,
) -> List[Dict]:
    """
    If a figure caption appears on the next page, stitch it to the prior figure.
    """
    figures = [elem for elem in elements if elem.get("type") == "figure"]
    if not figures or not all_dets:
        return elements

    try:
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
    except Exception as exc:
        logger.error(f"Unable to reopen PDF for figure caption stitching: {exc}")
        return elements

    page_cache: Dict[int, Image.Image] = {}
    used_following_ids: Set[Tuple[int, int]] = set()

    # Mark existing caption/title/text detections as used
    for elem in figures:
        for key in ("captions", "titles", "texts"):
            for seg in elem.get(key, []) or []:
                idx = seg.get("index")
                page_no = seg.get("page")
                if idx is None or page_no is None:
                    continue
                used_following_ids.add((page_no - 1, idx))

    for elem in figures:
        page_no = elem.get("page")
        bbox = elem.get("bbox_pixels")
        if page_no is None or bbox is None:
            continue

        current_idx = page_no - 1
        next_idx = current_idx + 1
        if next_idx >= len(all_dets):
            continue

        next_dets = all_dets[next_idx]
        if not next_dets:
            continue

        fig_width = bbox[2] - bbox[0]
        page_img = _render_pdf_page(pdf_doc, next_idx, scale, page_cache)
        if page_img is None:
            continue

        next_page_height = page_img.height
        max_top_allowed = min(
            CROSS_PAGE_CAPTION_THRESHOLDS["max_top_pixels"],
            int(next_page_height * CROSS_PAGE_CAPTION_THRESHOLDS["max_top_ratio"]),
        )

        sorted_next = sorted(
            [det for det in next_dets if det.get("bbox")],
            key=lambda det: det["bbox"][1],
        )

        caption_candidate: Optional[Tuple[Dict, int]] = None
        caption_candidates = []
        for det in sorted_next:
            if det.get("name") != "figure_caption":
                continue
            det_index = det.get("index")
            if det_index is None or (next_idx, det_index) in used_following_ids:
                continue

            det_bbox = det.get("bbox")
            if not det_bbox or det_bbox[1] > max_top_allowed:
                continue

            overlap = _horizontal_overlap_ratio(bbox, det_bbox)
            x_diff = abs(bbox[0] - det_bbox[0])
            width_diff = abs((bbox[2] - bbox[0]) - (det_bbox[2] - det_bbox[0]))

            if overlap < CROSS_PAGE_CAPTION_THRESHOLDS["min_overlap"]:
                if (
                    x_diff > CROSS_PAGE_CAPTION_THRESHOLDS["x_tol"]
                    or width_diff > CROSS_PAGE_CAPTION_THRESHOLDS["width_tol"]
                ):
                    continue

            score = width_diff + 0.5 * x_diff
            caption_candidates.append((score, det, det_index))

        if caption_candidates:
            caption_candidates.sort(key=lambda item: item[0])
            _, best_det, best_index = caption_candidates[0]
            caption_candidate = (best_det, best_index)

        title_candidate: Optional[Tuple[Dict, int]] = None
        title_texts: List[Dict] = []
        for idx_sorted, det in enumerate(sorted_next):
            if det.get("name") != "title":
                continue
            det_index = det.get("index")
            if det_index is None or (next_idx, det_index) in used_following_ids:
                continue

            det_bbox = det.get("bbox")
            if not det_bbox or det_bbox[1] > max_top_allowed:
                continue

            overlap = _horizontal_overlap_ratio(bbox, det_bbox)
            x_diff = abs(bbox[0] - det_bbox[0])
            if (
                overlap < TITLE_TEXT_ASSOCIATION["min_overlap"]
                and x_diff > CROSS_PAGE_CAPTION_THRESHOLDS["x_tol"]
            ):
                continue

            title_candidate = (det, det_index)
            title_texts = _collect_text_under_title_cross_page(
                det, sorted_next, idx_sorted, next_idx, used_following_ids
            )
            break

        if not caption_candidate and not title_candidate and not title_texts:
            continue

        figure_path = out_dir / elem["image_path"]
        if not figure_path.exists():
            continue

        figure_img = Image.open(figure_path)
        if figure_img.mode == "CMYK":
            figure_img = figure_img.convert("RGB")

        segments_added = False

        if caption_candidate:
            cap_det, cap_index = caption_candidate
            caption_crop = _crop_pdf_region(page_img, cap_det["bbox"])
            if caption_crop is not None:
                figure_img = _append_segment_image(
                    figure_img, caption_crop, resize_to_base=True
                )
                elem.setdefault("captions", [])
                elem["captions"].append(
                    {
                        "bbox": cap_det["bbox"],
                        "conf": cap_det.get("conf"),
                        "index": cap_index,
                        "source": cap_det.get("source"),
                        "page": next_idx + 1,
                    }
                )
                used_following_ids.add((next_idx, cap_index))
                segments_added = True

        if title_candidate:
            title_det, title_index = title_candidate
            title_crop = _crop_pdf_region(page_img, title_det["bbox"])
            if title_crop is not None:
                figure_img = _append_segment_image(figure_img, title_crop)
                elem.setdefault("titles", [])
                elem["titles"].append(
                    {
                        "bbox": title_det["bbox"],
                        "conf": title_det.get("conf"),
                        "index": title_index,
                        "source": title_det.get("source"),
                        "page": next_idx + 1,
                    }
                )
                used_following_ids.add((next_idx, title_index))
                segments_added = True

            for text_det in title_texts:
                text_index = text_det.get("index")
                text_crop = _crop_pdf_region(page_img, text_det["bbox"])
                if text_crop is None:
                    continue
                figure_img = _append_segment_image(figure_img, text_crop)
                elem.setdefault("texts", [])
                elem["texts"].append(
                    {
                        "bbox": text_det["bbox"],
                        "conf": text_det.get("conf"),
                        "index": text_index,
                        "source": text_det.get("source"),
                        "page": next_idx + 1,
                    }
                )
                if text_index is not None:
                    used_following_ids.add((next_idx, text_index))
                segments_added = True

        if not segments_added:
            continue

        figure_img.save(figure_path)
        elem["width"] = figure_img.width
        elem["height"] = figure_img.height

        span = elem.get("page_span")
        if span:
            if next_idx + 1 not in span:
                span.append(next_idx + 1)
        else:
            base_page = elem.get("page")
            new_span = [page for page in (base_page, next_idx + 1) if page is not None]
            elem["page_span"] = new_span

    pdf_doc.close()
    return elements


def _stitch_table_pair(
    base_elem: Dict,
    candidate_elem: Dict,
    out_dir: Path,
    merge_index: int,
    stitch_type: str,
) -> Optional[Dict]:
    """Stitch two table crops either vertically or horizontally."""
    base_img = _open_table_image(base_elem, out_dir)
    candidate_img = _open_table_image(candidate_elem, out_dir)
    if base_img is None or candidate_img is None:
        return None

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if stitch_type == "vertical":
        target_width = max(base_img.width, candidate_img.width)
        base_img = _pad_width(base_img, target_width)
        candidate_img = _pad_width(candidate_img, target_width)
        merged_height = base_img.height + candidate_img.height
        stitched = Image.new("RGB", (target_width, merged_height), color=(255, 255, 255))
        stitched.paste(base_img, (0, 0))
        stitched.paste(candidate_img, (0, base_img.height))
    else:
        target_height = max(base_img.height, candidate_img.height)
        base_img = _pad_height(base_img, target_height)
        candidate_img = _pad_height(candidate_img, target_height)
        merged_width = base_img.width + candidate_img.width
        stitched = Image.new("RGB", (merged_width, target_height), color=(255, 255, 255))
        stitched.paste(base_img, (0, 0))
        stitched.paste(candidate_img, (base_img.width, 0))

    merged_name = (
        f"page_{base_elem['page']}_to_{candidate_elem['page']}_"
        f"table_merged_{merge_index}.png"
    )
    merged_path = tables_dir / merged_name
    stitched.save(merged_path)

    # Remove original partial crops to avoid duplicates
    (out_dir / base_elem["image_path"]).unlink(missing_ok=True)
    (out_dir / candidate_elem["image_path"]).unlink(missing_ok=True)

    new_bbox = [
        min(base_elem["bbox_pixels"][0], candidate_elem["bbox_pixels"][0]),
        min(base_elem["bbox_pixels"][1], candidate_elem["bbox_pixels"][1]),
        max(base_elem["bbox_pixels"][2], candidate_elem["bbox_pixels"][2]),
        max(base_elem["bbox_pixels"][3], candidate_elem["bbox_pixels"][3]),
    ]

    merged_elem = base_elem.copy()
    merged_elem["page_span"] = [base_elem["page"], candidate_elem["page"]]
    merged_elem["box_refs"] = [
        {"page": base_elem["page"], "image_path": base_elem["image_path"]},
        {"page": candidate_elem["page"], "image_path": candidate_elem["image_path"]},
    ]
    merged_elem["bbox_pixels"] = new_bbox
    merged_elem["image_path"] = str(merged_path.relative_to(out_dir))
    merged_elem["width"] = stitched.width
    merged_elem["height"] = stitched.height
    merged_elem["page_height"] = stitched.height
    merged_elem["conf"] = min(
        base_elem.get("conf", 1.0), candidate_elem.get("conf", 1.0)
    )
    return merged_elem


def merge_spanning_tables(elements: List[Dict], out_dir: Path) -> List[Dict]:
    """
    Stitch table crops that continue across adjacent pages using the heuristic
    from the legacy OpenCV-based extractor.
    """
    if not elements:
        return elements

    tables_by_page: Dict[int, List[Dict]] = {}
    non_tables: List[Dict] = []

    for elem in elements:
        if elem.get("type") != "table":
            non_tables.append(elem)
            continue
        page = elem.get("page")
        if not isinstance(page, int):
            non_tables.append(elem)
            continue
        tables_by_page.setdefault(page, []).append(elem)

    merged_results: List[Dict] = []
    used_next: Dict[int, set[int]] = {}
    merge_counter = 0

    for page in sorted(tables_by_page.keys()):
        current_tables = tables_by_page.get(page, [])
        next_page_tables = tables_by_page.get(page + 1, [])
        next_used_indices = used_next.get(page + 1, set())
        current_used_indices = used_next.get(page, set())

        for idx_current, table_elem in enumerate(current_tables):
            if idx_current in current_used_indices:
                continue

            if not next_page_tables:
                merged_results.append(table_elem)
                continue

            x, y, w, h = _bbox_to_rect(table_elem["bbox_pixels"])
            matched = False

            for idx, candidate in enumerate(next_page_tables):
                if idx in next_used_indices:
                    continue
                if candidate.get("type") != "table":
                    continue

                cx, cy, cw, ch = _bbox_to_rect(candidate["bbox_pixels"])

                vertical_match = (
                    abs(x - cx) <= TABLE_STITCH_TOLERANCES["x_tol"]
                    and abs((x + w) - (cx + cw)) <= TABLE_STITCH_TOLERANCES["width_tol"]
                )
                horizontal_match = (
                    abs(y - cy) <= TABLE_STITCH_TOLERANCES["y_tol"]
                    and abs((y + h) - (cy + ch))
                    <= TABLE_STITCH_TOLERANCES["height_tol"]
                )

                stitch_type = "vertical" if vertical_match else None
                if not stitch_type and horizontal_match:
                    stitch_type = "horizontal"

                if not stitch_type:
                    continue

                merge_counter += 1
                merged_elem = _stitch_table_pair(
                    table_elem, candidate, out_dir, merge_counter, stitch_type
                )
                if merged_elem is None:
                    continue

                merged_results.append(merged_elem)
                next_used_indices.add(idx)
                matched = True
                break

            if not matched:
                merged_results.append(table_elem)

        used_next[page + 1] = next_used_indices

    merged_results.extend(non_tables)
    return merged_results



# ----------------------------------------------------------------------
# Draw layout boxes on the original PDF
# ----------------------------------------------------------------------
def draw_layout_pdf(pdf_bytes: bytes, all_dets: List[List[dict]],
                    scale: float, out_path: Path):
    """Annotate PDF with semi-transparent bounding boxes and labels."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_no, dets in enumerate(all_dets):
        page = doc[page_no]

        for d in dets:
            rgb = CLASS_COLORS.get(d["name"], (0, 0, 0))
            rect = fitz.Rect([c / scale for c in d["bbox"]])

            border_color = [c / 255 for c in rgb]
            fill_color = [c / 255 for c in rgb]
            fill_opacity = 0.15
            border_width = 1.5

            page.draw_rect(
                rect,
                color=border_color,
                fill=fill_color,
                width=border_width,
                overlay=True,
                fill_opacity=fill_opacity
            )

            label = f"{d['name']} {d['conf']:.2f}"
            if d.get("source"):
                label += f" [{d['source'][0].upper()}]"

            text_bg = fitz.Rect(rect.x0, rect.y0 - 10, rect.x0 + 60, rect.y0)
            page.draw_rect(text_bg, color=None, fill=(1, 1, 1, 0.6), overlay=True)

            page.insert_text(
                (rect.x0 + 2, rect.y0 - 8),
                label,
                fontsize=6.5,
                color=border_color,
                overlay=True
            )

    doc.save(str(out_path))
    doc.close()

# ----------------------------------------------------------------------
# Process a single PDF Page (for parallel execution)
# ----------------------------------------------------------------------
def process_page(task_data: Tuple[int, bytes, float, Path, str]) -> Optional[Tuple[int, List[dict], List[dict]]]:
    """
    Process a single page of a PDF in a worker process.
    Returns: (page_number, detections, elements) or None on failure
    """
    pno, pdf_bytes, scale, out_dir, pdf_name = task_data
    
    if _shutdown_requested:
        return None
    
    pdf_pdfium = None
    try:
        pdf_pdfium = pdfium.PdfDocument(pdf_bytes)
        
        page = pdf_pdfium[pno]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()

        dets = detect_page(pil)
        elements = save_layout_elements(pil, pno, dets, out_dir)
        
        page_figures = len([d for d in dets if d['name'] == 'figure'])
        page_tables = len([d for d in dets if d['name'] == 'table'])
        logger.info(f"  [{pdf_name}] Page {pno + 1}: {page_figures} figs, {page_tables} tables")

        page.close()
        pdf_pdfium.close()
        
        return (pno, dets, elements)

    except Exception as e:
        logger.error(f"Failed to process page {pno + 1} of {pdf_name}: {e}")
        if pdf_pdfium:
            pdf_pdfium.close()
        return None

# ----------------------------------------------------------------------
# Process a full PDF using the persistent worker pool
# ----------------------------------------------------------------------
def process_pdf_with_pool(
    pdf_path: Path,
    out_dir: Path,
    pool: Optional[Pool] = None,
    *,
    extract_images: bool = True,
    extract_markdown: bool = True,
):
    """
    Main processing pipeline for a PDF file.
    If pool is provided, uses it. Otherwise processes serially.
    """
    
    if _shutdown_requested:
        logger.warning(f"Skipping {pdf_path.name} due to shutdown request")
        return
    
    stem = pdf_path.stem
    logger.info(f"Processing {pdf_path.name}")

    pdf_bytes = pdf_path.read_bytes()
    
    doc = None
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        page_count = len(doc)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path.name}: {e}. Skipping.")
        return
    finally:
        if doc is not None:
            doc.close()

    scale = 2.0
    all_elements: List[Dict] = []
    filtered_dets: List[List[dict]] = []

    if extract_images:
        all_dets: List[Optional[List[dict]]] = [None] * page_count

        if pool is not None and USE_MULTIPROCESSING:
            logger.info(f"  Using worker pool for {page_count} pages...")

            tasks = [
                (pno, pdf_bytes, scale, out_dir, pdf_path.name)
                for pno in range(page_count)
            ]

            try:
                results = pool.map(process_page, tasks)

                for res in results:
                    if res:
                        pno, dets, elements = res
                        all_dets[pno] = dets
                        all_elements.extend(elements)

            except KeyboardInterrupt:
                logger.warning("Processing interrupted during parallel execution")
                raise

        else:
            logger.info("Using serial processing...")

            try:
                pdf_pdfium = pdfium.PdfDocument(pdf_bytes)

                for pno in range(page_count):
                    if _shutdown_requested:
                        logger.warning(
                            f"Stopping at page {pno + 1}/{page_count} due to shutdown request"
                        )
                        break

                    try:
                        logger.info(f"  Processing page {pno + 1}/{page_count}")

                        page = pdf_pdfium[pno]
                        bitmap = page.render(scale=scale)
                        pil = bitmap.to_pil()

                        dets = detect_page(pil)
                        all_dets[pno] = dets

                        elements = save_layout_elements(pil, pno, dets, out_dir)
                        all_elements.extend(elements)

                        page_figures = len([d for d in dets if d["name"] == "figure"])
                        page_tables = len([d for d in dets if d["name"] == "table"])
                        logger.info(
                            f"    Found {page_figures} figures and {page_tables} tables"
                        )

                        page.close()

                    except Exception as e:
                        logger.error(f"Failed to process page {pno + 1}: {e}. Skipping page.")

                pdf_pdfium.close()

            except Exception as e:
                logger.error(f"Fatal error processing {pdf_path.name}: {e}")
                if "pdf_pdfium" in locals() and pdf_pdfium:
                    pdf_pdfium.close()
                return

        dets_per_page: List[Optional[List[Dict[str, Any]]]] = [
            det if det is not None else None for det in all_dets
        ]

        filtered_dets = [d for d in all_dets if d is not None]

        if all_elements:
            all_elements = merge_spanning_tables(all_elements, out_dir)
            all_elements = attach_cross_page_figure_captions(
                all_elements, dets_per_page, pdf_bytes, out_dir, scale
            )

        if all_elements:
            content_list_path = out_dir / f"{stem}_content_list.json"
            with open(content_list_path, "w", encoding="utf-8") as f:
                json.dump(all_elements, f, ensure_ascii=False, indent=4)
            logger.info(f"  Saved {len(all_elements)} elements to JSON")

        if filtered_dets:
            draw_layout_pdf(
                pdf_bytes, filtered_dets, scale, out_dir / f"{stem}_layout.pdf"
            )
            logger.info("  Generated annotated PDF")
        else:
            logger.warning(f"No detections found for {stem}. Skipping layout PDF.")

    else:
        logger.info("  Image extraction skipped per configuration.")

    markdown_path = None
    if extract_markdown:
        markdown_path = write_markdown_document(pdf_path, out_dir)
        if markdown_path is None:
            logger.warning(f"  Markdown extraction yielded no content for {stem}.")

    if _shutdown_requested:
        logger.warning(f"⚠️  Partial results saved for {stem} → {out_dir}")
    else:
        if extract_images:
            logger.success(
                f"✓ {stem} → {out_dir} ({len(all_elements)} elements extracted)"
            )
        else:
            logger.success(f"✓ {stem} → {out_dir} (image extraction skipped)")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Important for multiprocessing on Windows/macOS
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    INPUT_DIR = Path("./pdfs")
    OUTPUT_DIR = Path("./output")

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in ./pdfs")
        logger.info("Please add PDF files to the ./pdfs directory")
        logger.info("The script will exit gracefully. No errors occurred.")
        sys.exit(0)

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info(f"Settings: MODEL_SIZE={MODEL_SIZE}, CONF={CONF_THRESHOLD}")
    
    # Determine worker count
    total_cpus = cpu_count()
    if NUM_WORKERS is None:
        num_workers = max(1, total_cpus - 1)
    else:
        num_workers = max(1, min(NUM_WORKERS, total_cpus))
    
    # Decide whether to use multiprocessing
    use_pool = USE_MULTIPROCESSING and DEVICE == "cpu" and total_cpus >= 4
    
    if use_pool:
        logger.info(f"🚀 Creating persistent worker pool with {num_workers} workers...")
    else:
        if not USE_MULTIPROCESSING:
            logger.info("Multiprocessing disabled by configuration")
        elif DEVICE != "cpu":
            logger.info(f"Using serial GPU processing (device: {DEVICE})")
        else:
            logger.info(f"Using serial CPU processing (CPU count {total_cpus} too low)")

    pool = None
    try:
        # Create persistent pool ONCE for all PDFs
        if use_pool:
            pool = Pool(processes=num_workers, initializer=init_worker)
            logger.success(f"✓ Worker pool ready with {num_workers} workers\n")
        else:
            # Load model in main process for serial execution
            logger.info("Initializing model in main process...")
            get_model()
            logger.success(f"✓ Model loaded (device: {DEVICE})\n")

        # Process all PDFs using the same pool
        for i, pdf_path in enumerate(pdf_files, 1):
            if _shutdown_requested:
                logger.warning(f"\nShutdown requested. Processed {i-1}/{len(pdf_files)} files.")
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📄 File {i}/{len(pdf_files)}: {pdf_path.name}")
            logger.info(f"{'='*60}")
            
            sub_out = OUTPUT_DIR / pdf_path.stem
            os.makedirs(sub_out, exist_ok=True)
            
            try:
                process_pdf_with_pool(pdf_path, sub_out, pool)
            except KeyboardInterrupt:
                logger.warning(f"\nInterrupted while processing {pdf_path.name}")
                break
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                if _shutdown_requested:
                    break
                logger.info("Continuing with next file...")
                continue

        if _shutdown_requested:
            logger.warning(f"\n⚠️  Processing interrupted. Partial results saved in {OUTPUT_DIR}")
        else:
            logger.success(f"\n✨ All done! Results are in {OUTPUT_DIR}")
            
    except KeyboardInterrupt:
        logger.error("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Clean up pool if it exists
        if pool is not None:
            logger.info("\n🧹 Shutting down worker pool...")
            pool.close()
            pool.join()
            logger.success("✓ Worker pool closed cleanly")