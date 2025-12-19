import json
import os
import shutil
import shutil
import threading
import uuid
import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re
import gradio as gr
# from werkzeug.utils import secure_filename # Removed dependency
import torch

import main as extractor
from loguru import logger

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # Not strictly enforced by FastAPI by default, but good to know
UPLOAD_FOLDER = Path('./uploads')
OUTPUT_FOLDER = Path('./output')

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Global model instance
_model = None
_progress_tracker: Dict[str, Dict] = {}
_progress_lock = threading.RLock()
# Global process pool
_pool = None


def secure_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and special chars.
    Simplistic implementation to replace werkzeug.
    """
    filename = Path(filename).name
    # Keep only alphanumeric, dots, hyphens, and underscores
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename


def get_device_info() -> Dict[str, Any]:
    """Get information about GPU/CPU availability."""
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    
    info = {
        "device": device,
        "cuda_available": cuda_available,
        "device_name": None,
        "device_count": 0,
    }
    
    if cuda_available:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    
    return info

def load_model_once():
    """Load the model once and cache it."""
    global _model
    if _model is None:
        logger.info("Loading DocLayout-YOLO model...")
        _model = extractor.get_model()
        logger.info("Model loaded successfully")
    return _model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Life span context manager for startup and shutdown events.
    Initializes the multiprocessing pool for non-blocking CPU tasks.
    """
    global _pool
    logger.info("Starting up PDF Layout Extractor...")
    
    # Configure multiprocessing for PyTorch/CUDA
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set
        
    # Initialize worker pool
    try:
        # On Hugging Face Spaces (Free), we usually have 2 vCPUs. 
        # cpu_count() might return the HOST's count (e.g. 16), leading to too many spawn attempts.
        # Let's cap it safely.
        import os
        
        # logical_cpu_count = multiprocessing.cpu_count()
        # workers = max(1, min(logical_cpu_count - 1, 4)) # Cap at 4 for safety
        
        # Simpler default for stability:
        workers = 2 
        
        logger.info(f"Initializing background process pool with {workers} workers...")
        _pool = multiprocessing.Pool(processes=workers, initializer=extractor.init_worker)
    except Exception as e:
        logger.error(f"Failed to initialize pool: {e}")
        # non-fatal, will fallback to serial? 
        # actually if pool is None, app might error if we rely on it.
        # But we'll handle it.
        pass

    yield
    
    # Shutdown
    logger.info("Shutting down PDF Layout Extractor...")
    if _pool:
        _pool.close()
        _pool.join()

app = FastAPI(
    title="PDF Layout Extractor API",
    description="A polished API for extracting layout information (text, tables, figures) from PDFs using DocLayout-YOLO.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
# Mount Output as Static for easy access to generated images/PDFs
app.mount("/output", StaticFiles(directory="output"), name="output")


# --------------------------------------------------------------------------------
# Pydantic Models for Response Documentation
# --------------------------------------------------------------------------------

class DeviceInfo(BaseModel):
    device: str = Field(..., description="Compute device being used (e.g., 'cuda' or 'cpu').")
    cuda_available: bool = Field(..., description="Whether CUDA GPU acceleration is available.")
    device_name: Optional[str] = Field(None, description="Name of the GPU if available.")
    device_count: int = Field(..., description="Number of GPU devices detected.")

class TaskStartResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the background processing task.")
    message: str = Field(..., description="Status message confirming start.")
    total_files: int = Field(..., description="Number of PDF files accepted for processing.")

class ProcessingResult(BaseModel):
    filename: str = Field(..., description="Name of the processed file.")
    stem: Optional[str] = Field(None, description="Filename without extension.")
    output_dir: Optional[str] = Field(None, description="Relative path to the output directory.")
    figures_count: Optional[int] = Field(0, description="Total figures detected.")
    tables_count: Optional[int] = Field(0, description="Total tables detected.")
    elements_count: Optional[int] = Field(0, description="Total layout elements (text, tables, figures).")
    annotated_pdf: Optional[str] = Field(None, description="Path to the PDF with layout bounding boxes drawn.")
    markdown_path: Optional[str] = Field(None, description="Path to the extracted markdown file.")
    # Extended URLs
    annotated_pdf_url: Optional[str] = Field(None, description="Full URL to access the annotated PDF.")
    markdown_url: Optional[str] = Field(None, description="Full URL to access the extracted markdown.")
    figure_urls: Optional[List[Dict[str, Any]]] = Field(None, description="List of URLs for extracted figure images.")
    table_urls: Optional[List[Dict[str, Any]]] = Field(None, description="List of URLs for extracted table images.")
    error: Optional[str] = Field(None, description="Error message if processing failed.")

class ExtractionMode(str, Enum):
    images = "images"
    markdown = "markdown"
    both = "both"

class ProgressResponse(BaseModel):
    status: str = Field(..., description="Current status of the task (e.g., 'processing', 'completed').")
    progress: int = Field(..., description="Overall progress percentage (0-100).")
    message: str = Field(..., description="Current status message.")
    results: List[ProcessingResult] = Field([], description="List of results for processed files.")
    file_progress: Optional[Dict[str, int]] = Field(None, description="Progress percentage per file.")

class PDFInfo(BaseModel):
    stem: str = Field(..., description="Unique identifier/stem of the PDF.")
    output_dir: str = Field(..., description="Directory where results are stored.")

class PDFListResponse(BaseModel):
    pdfs: List[PDFInfo] = Field(..., description="List of processed PDFs available on the server.")

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def _update_task_progress(task_id: str, filename: str, file_progress: int, message: str):
    """Update progress for a specific file and calculate overall progress."""
    with _progress_lock:
        if task_id not in _progress_tracker:
            return
        
        # Update file-specific progress
        if 'file_progress' not in _progress_tracker[task_id]:
            _progress_tracker[task_id]['file_progress'] = {}
        _progress_tracker[task_id]['file_progress'][filename] = file_progress
        
        # Calculate overall progress (average of all files)
        file_progresses = _progress_tracker[task_id]['file_progress']
        if file_progresses:
            total_progress = sum(file_progresses.values()) / len(file_progresses)
            _progress_tracker[task_id]['progress'] = int(total_progress)
        
        _progress_tracker[task_id]['message'] = message

def process_file_background_task(task_id: str, file_data: bytes, filename: str, extraction_mode: str):
    """
    Process a single file in the background (runs in a thread pool inside FastAPI/Starlette).
    Note: For heavy CPU/GPU tasks, prefer running in a separate process or queue (like Celery),
    but consistent with the request to 'use FastAPI' and the previous design, this is fine
    since `fastapi.BackgroundTasks` runs in a thread pool.
    """
    filename = secure_filename(filename)
    
    try:
        _update_task_progress(task_id, filename, 5, f'Processing {filename}...')
        
        stem = Path(filename).stem
        include_images = extraction_mode != 'markdown'
        include_markdown = extraction_mode != 'images'
        
        # Ensure upload directory exists
        upload_dir = UPLOAD_FOLDER
        upload_path = upload_dir / filename
        upload_path.write_bytes(file_data)
        
        _update_task_progress(task_id, filename, 15, f'Saved {filename}, preparing output...')
        
        # Prepare output directory
        output_dir = OUTPUT_FOLDER / stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy PDF to output directory
        pdf_path = output_dir / filename
        # shutil.copy caused permissions issues in some envs, renaming/moving is safer if fresh upload
        # But here we might want to keep the original in uploads? 
        # The original code did `upload_path.rename(pdf_path)`, so let's stick to that semantics:
        # Move from temp upload to output dir
        if pdf_path.exists():
            pdf_path.unlink()
        upload_path.rename(pdf_path)
        
        _update_task_progress(task_id, filename, 25, f'Loading model and processing {filename}...')
        
        # Process PDF
        # Enable multiprocessing to release GIL and avoid blocking the event loop
        extractor.USE_MULTIPROCESSING = True 
        logger.info(f"Processing {filename} (images={include_images}, markdown={include_markdown})")
        
        # Note: When using a pool, we don't strictly need to load the model in THIS process
        # unless we fallback to serial. 
        # But 'init_worker' loaded it in workers.
        
        _update_task_progress(task_id, filename, 30, f'Extracting content from {filename}...')
        
        # Use the global pool
        # If _pool is None (initialization failed), main.py will fallback to serial (blocking this thread, but working)
        extractor.process_pdf_with_pool(
            pdf_path,
            output_dir,
            pool=_pool, 
            extract_images=include_images,
            extract_markdown=include_markdown,
        )
        
        _update_task_progress(task_id, filename, 85, f'Collecting results for {filename}...')
        
        # Collect results
        json_path = output_dir / f"{stem}_content_list.json"
        elements = []
        if include_images and json_path.exists():
            text_content = json_path.read_text(encoding='utf-8')
            if text_content.strip():
                elements = json.loads(text_content)
        
        annotated_pdf = None
        if include_images:
            candidate_pdf = output_dir / f"{stem}_layout.pdf"
            if candidate_pdf.exists():
                annotated_pdf = str(candidate_pdf.relative_to(OUTPUT_FOLDER))
        
        markdown_path = None
        if include_markdown:
            candidate_md = output_dir / f"{stem}.md"
            if candidate_md.exists():
                markdown_path = str(candidate_md.relative_to(OUTPUT_FOLDER))
        
        figures = [e for e in elements if e.get('type') == 'figure']
        tables = [e for e in elements if e.get('type') == 'table']
        
        result = {
            'filename': filename,
            'stem': stem,
            'output_dir': str(output_dir.relative_to(OUTPUT_FOLDER)),
            'figures_count': len(figures),
            'tables_count': len(tables),
            'elements_count': len(elements),
            'annotated_pdf': annotated_pdf,
            'markdown_path': markdown_path,
            'include_images': include_images,
            'include_markdown': include_markdown,
        }
        
        with _progress_lock:
            if 'file_progress' not in _progress_tracker[task_id]:
                _progress_tracker[task_id]['file_progress'] = {}
            _progress_tracker[task_id]['file_progress'][filename] = 100
            
            # Recalculate total
            file_progresses = _progress_tracker[task_id]['file_progress']
            if file_progresses:
                total_prog = sum(file_progresses.values()) / len(file_progresses)
                _progress_tracker[task_id]['progress'] = int(total_prog)
            
            _progress_tracker[task_id]['results'].append(result)
            _progress_tracker[task_id]['message'] = f'Completed processing {filename}'
            
            # Check completion
            total_files = _progress_tracker[task_id].get('total_files', 1)
            completed_count = len([r for r in _progress_tracker[task_id]['results'] if 'error' not in r])
            error_count = len([r for r in _progress_tracker[task_id]['results'] if 'error' in r])
            
            if completed_count + error_count >= total_files:
                _progress_tracker[task_id]['status'] = 'completed'
                _progress_tracker[task_id]['progress'] = 100
                _progress_tracker[task_id]['message'] = f'All {total_files} file(s) processed.'

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        with _progress_lock:
            _progress_tracker[task_id]['results'].append({
                'filename': filename,
                'error': str(e)
            })
            # Check if this was the last file
            total_files = _progress_tracker[task_id].get('total_files', 1)
            if len(_progress_tracker[task_id]['results']) >= total_files:
                _progress_tracker[task_id]['status'] = 'completed' # Mark done even if error, so frontend stops polling
                _progress_tracker[task_id]['message'] = f'Finished with errors.'


# --------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------

@app.get("/api/docs", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def api_docs_redirect():
    """Redirect legacy /api/docs to Swagger UI."""
    return HTMLResponse(
        """
        <html>
            <head>
                <meta http-equiv="refresh" content="0; url=/docs" />
            </head>
            <body>
                <p>Redirecting to <a href="/docs">/docs</a>...</p>
            </body>
        </html>
        """
    )


@app.get("/api/device-info", response_model=DeviceInfo, tags=["System"])
async def device_info_endpoint():
    """Get information about the processing device (CPU/GPU)."""
    return get_device_info()


@app.post("/api/upload", response_model=TaskStartResponse, tags=["Processing"])
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    extraction_mode: ExtractionMode = Form(ExtractionMode.images, description="Select extraction mode: 'images' (figures/tables), 'markdown' (text), or 'both'.")
):
    """
    Upload one or more PDF files for background processing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No valid PDF files selected")

    task_id = str(uuid.uuid4())
    
    with _progress_lock:
        _progress_tracker[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting upload...',
            'results': [],
            'total_files': len(pdf_files)
        }
    
    # Read files into memory to pass to background task (UploadFile is a stream)
    # Be careful with RAM here for huge files. If too big, save to temp disk first.
    # Given the original code read into RAM, we'll do the same for consistency but simpler.
    for file in pdf_files:
        content = await file.read()
        background_tasks.add_task(
            process_file_background_task, 
            task_id, 
            content, 
            file.filename, 
            extraction_mode
        )
    
    return {
        "task_id": task_id,
        "message": "Processing started",
        "total_files": len(pdf_files)
    }


@app.get("/api/progress/{task_id}", response_model=ProgressResponse, tags=["Processing"])
async def get_progress(task_id: str, request: Request):
    """Check the progress of a processing task."""
    with _progress_lock:
        progress = _progress_tracker.get(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Deep copy to modify for response (adding URLs) without changing state
        # Or just build the response object.
        # Since we are adding computed URLs, we shouldn't modify the stored state every time.
        response_data = progress.copy()
        
        # Use request.base_url for absolute URLs
        base_url = str(request.base_url).rstrip('/')
        if 'hf.space' in base_url or request.headers.get("x-forwarded-proto") == "https":
            base_url = base_url.replace("http://", "https://")

        # Process results to add URLs
        results_with_urls = []
        for res in response_data.get('results', []):
            res_copy = res.copy()
            
            # Helper to make url
            def make_url(rel_path):
                if not rel_path: return None
                # Clean windows paths to forward slashes for URLs
                clean_path = str(rel_path).replace('\\', '/')
                return f"{base_url}/output/{clean_path}"

            res_copy['annotated_pdf_url'] = make_url(res.get('annotated_pdf'))
            res_copy['markdown_url'] = make_url(res.get('markdown_path'))
            
            # Figures and Tables URLs need to be discovered from disk if not stored
            # The original code loaded JSON every time. That's a bit heavy but ensures freshness.
            # Let's try to do it if stem is present.
            stem = res.get('stem')
            if stem:
                output_dir = OUTPUT_FOLDER / stem
                if output_dir.exists():
                    json_files = list(output_dir.glob('*_content_list.json'))
                    if json_files:
                        try:
                            elements = json.loads(json_files[0].read_text(encoding='utf-8'))
                            figures = [e for e in elements if e.get('type') == 'figure']
                            tables = [e for e in elements if e.get('type') == 'table']
                            
                            fig_urls = []
                            for fig in figures:
                                if fig.get('image_path'):
                                    path = Path(fig['image_path']) # relative to unique output folder usually?
                                    # Actually in main.py it saves relative to out_dir
                                    # so image_path is like "figures/page_1_fig_0.png"
                                    # We need relative to "output" folder for URL
                                    # output_dir is "output/stem_timestamp" 
                                    # so full path is "output/stem_timestamp/figures/..."
                                    # The URL mount is /output/ -> output/
                                    
                                    # "image_path" in JSON is relative to the specific STEM folder (implied by main.py logic)
                                    # Wait, main.py says: "image_path": str(path_template.relative_to(out_dir))
                                    # So yes, it is "figures/..."
                                    
                                    full_rel_path = f"{stem}/{fig['image_path']}"
                                    fig_urls.append({
                                        "page": fig.get('page'),
                                        "url": make_url(full_rel_path),
                                        "path": full_rel_path
                                    })
                            res_copy['figure_urls'] = fig_urls
                            
                            tab_urls = []
                            for tab in tables:
                                if tab.get('image_path'):
                                    full_rel_path = f"{stem}/{tab['image_path']}"
                                    tab_urls.append({
                                        "page": tab.get('page'),
                                        "url": make_url(full_rel_path),
                                        "path": full_rel_path
                                    })
                            res_copy['table_urls'] = tab_urls

                        except Exception as e:
                            logger.error(f"Error reading details for {stem}: {e}")
            
            results_with_urls.append(res_copy)
        
        response_data['results'] = results_with_urls
        return response_data


@app.get("/api/pdf-list", response_model=PDFListResponse, tags=["Retrieval"])
async def pdf_list():
    """List previously processed PDFs."""
    output_dir = OUTPUT_FOLDER
    pdfs = []
    
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                # Check for indicators of success
                if list(item.glob('*_content_list.json')) or list(item.glob('*.md')):
                    pdfs.append({
                        'stem': item.name,
                        'output_dir': item.name # returning the name as relative dir
                    })
    return {'pdfs': pdfs}


@app.get("/api/pdf-details/{pdf_stem}", tags=["Retrieval"])
async def pdf_details(pdf_stem: str, request: Request):
    """Get detailed information about a processed PDF."""
    output_dir = OUTPUT_FOLDER / pdf_stem
    
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    base_url = str(request.base_url).rstrip('/')
    if 'hf.space' in base_url or request.headers.get("x-forwarded-proto") == "https":
        base_url = base_url.replace("http://", "https://")

    def make_url(rel_path):
        if not rel_path: return None
        clean_path = str(rel_path).replace('\\', '/')
        return f"{base_url}/output/{clean_path}"

    # Load content list
    json_files = list(output_dir.glob('*_content_list.json'))
    elements = []
    if json_files:
        elements = json.loads(json_files[0].read_text(encoding='utf-8'))
    
    figures = [e for e in elements if e.get('type') == 'figure']
    tables = [e for e in elements if e.get('type') == 'table']
    
    # PDF Layout
    annotated_pdf = None
    pdf_files = list(output_dir.glob('*_layout.pdf'))
    if pdf_files:
        annotated_pdf = f"{pdf_stem}/{pdf_files[0].name}"
    
    # Markdown
    markdown_path = None
    md_files = list(output_dir.glob('*.md'))
    if md_files:
        markdown_path = f"{pdf_stem}/{md_files[0].name}"
        
    # Image lists
    figure_images = []
    fig_dir = output_dir / 'figures'
    if fig_dir.exists():
        figure_images = [f"{pdf_stem}/figures/{f.name}" for f in sorted(fig_dir.glob('*.png'))]
        
    table_images = []
    tab_dir = output_dir / 'tables'
    if tab_dir.exists():
        table_images = [f"{pdf_stem}/tables/{f.name}" for f in sorted(tab_dir.glob('*.png'))]

    return {
        'stem': pdf_stem,
        'figures': figures,
        'tables': tables,
        'figures_count': len(figures),
        'tables_count': len(tables),
        'elements_count': len(elements),
        'annotated_pdf': annotated_pdf,
        'markdown_path': markdown_path,
        'figure_images': figure_images,
        'table_images': table_images,
        'urls': {
            'annotated_pdf': make_url(annotated_pdf),
            'markdown': make_url(markdown_path),
            'figures': [make_url(img) for img in figure_images],
            'tables': [make_url(img) for img in table_images],
        }
    }


@app.post("/api/predict", tags=["Legacy"], include_in_schema=True)
async def predict(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Direct API endpoint for extracting text/tables/figures from a single PDF.
    Waits for completion and returns JSON result.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    # Create unique output directory
    filename = secure_filename(file.filename)
    stem = Path(filename).stem
    unique_id = f"{stem}_{int(time.time())}"
    output_dir = OUTPUT_FOLDER / unique_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    pdf_path = output_dir / filename
    content = await file.read()
    pdf_path.write_bytes(content)
    
    try:
        # Load model logic (sync call to stay simple for this endpoint)
        load_model_once()
        extractor.USE_MULTIPROCESSING = False
        
        # Process
        extractor.process_pdf_with_pool(
            pdf_path,
            output_dir,
            pool=None,
            extract_images=True,
            extract_markdown=True,
        )
        
        # Build Result
        base_url = str(request.base_url).rstrip('/')
        if 'hf.space' in base_url or request.headers.get("x-forwarded-proto") == "https":
            base_url = base_url.replace("http://", "https://")

        def make_url(rel_path):
             return f"{base_url}/output/{unique_id}/{rel_path}"

        result = {
            "status": "success",
            "filename": filename,
            "text": "",
            "tables": [],
            "figures": [],
            "summary": {}
        }
        
        # Text
        md_path = output_dir / f"{stem}.md"
        if md_path.exists():
            result['text'] = md_path.read_text(encoding='utf-8')
            
        # JSON content
        json_path = output_dir / f"{stem}_content_list.json"
        if json_path.exists():
            elements = json.loads(json_path.read_text(encoding='utf-8'))
            
            figures = [e for e in elements if e.get('type') == 'figure']
            result['figures'] = [{
                **fig, 
                'image_url': make_url(fig.get('image_path')) if fig.get('image_path') else None
            } for fig in figures]
            
            tables = [e for e in elements if e.get('type') == 'table']
            result['tables'] = [{
                **tab, 
                'image_url': make_url(tab.get('image_path')) if tab.get('image_path') else None
            } for tab in tables]
            
            result['summary'] = {
                'figures_count': len(figures),
                'tables_count': len(tables),
                'elements_count': len(elements)
            }
            
        return result

    except Exception as e:
        logger.error(f"Error in predict: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/delete", tags=["Processing"])
async def delete_pdf(stem: str = Form(...)):
    """Delete a processed PDF and its output directory."""
    if not stem:
        raise HTTPException(status_code=400, detail="Missing stem")
    
    # Resolve output directory safely
    output_root = OUTPUT_FOLDER.resolve()
    target_dir = (output_root / stem).resolve()
    
    # Prevent path traversal
    if output_root not in target_dir.parents and target_dir != output_root:
         raise HTTPException(status_code=400, detail="Invalid stem path")
         
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="Not found")
        
    try:
        shutil.rmtree(target_dir)
        return {"status": "success", "message": f"Deleted {stem}"}
    except Exception as e:
        # Try to fix read-only files (common on Windows)
        try:
            import stat
            def on_rm_error(func, path, exc_info):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(target_dir, onerror=on_rm_error)
            return {"status": "success", "message": f"Deleted {stem}"}
        except Exception as e2:
            logger.error(f"Error deleting {stem}: {e2}")
            raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e2)}")


# --------------------------------------------------------------------------------
# Gradio Interface
# --------------------------------------------------------------------------------

def gradio_process(pdf_file, mode_str):
    """
    Wrapper for Gradio to call the extractor logic.
    """
    if pdf_file is None:
        return None, None, None, "No file uploaded."
        
    try:
        # Create unique directory
        filename = secure_filename(Path(pdf_file.name).name)
        stem = Path(filename).stem
        unique_id = f"{stem}_{int(time.time())}"
        output_dir = OUTPUT_FOLDER / unique_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        dest_path = output_dir / filename
        shutil.copy(pdf_file.name, dest_path)
        
        # Determine flags
        include_images = (mode_str != "markdown")
        include_markdown = (mode_str != "images")
        
        # Process using the multiprocessing pool for speed
        # The global pool is already initialized in lifespan
        extractor.USE_MULTIPROCESSING = True
        
        extractor.process_pdf_with_pool(
            dest_path,
            output_dir,
            pool=_pool,  # Use the global pool instead of None
            extract_images=include_images,
            extract_markdown=include_markdown
        )
        
        # Collect outputs
        md_text = ""
        md_path = output_dir / f"{stem}.md"
        if md_path.exists():
            md_text = md_path.read_text(encoding='utf-8')
            
        annotated_pdf = None
        pdf_layout_path = output_dir / f"{stem}_layout.pdf"
        if pdf_layout_path.exists():
            annotated_pdf = str(pdf_layout_path)
            
        gallery = []
        if include_images:
            fig_dir = output_dir / 'figures'
            if fig_dir.exists():
                gallery.extend([str(p) for p in fig_dir.glob('*.png')])
            tab_dir = output_dir / 'tables'
            if tab_dir.exists():
                gallery.extend([str(p) for p in tab_dir.glob('*.png')])
                
        return md_text, gallery, annotated_pdf, f"Processed {filename} successfully."

    except Exception as e:
        logger.error(f"Gradio Error: {e}")
        return str(e), None, None, f"Error: {e}"

# Define Gradio App
with gr.Blocks(title="PDF Layout Extractor") as demo:
    gr.Markdown("# PDF Layout Extractor")
    gr.Markdown("Upload a PDF to extract text (Markdown), figures, tables, and visualization.")
    
    with gr.Row():
        with gr.Column():
            input_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
            mode_input = gr.Radio(["both", "images", "markdown"], label="Extraction Mode", value="both")
            process_btn = gr.Button("Extract Layout", variant="primary")
            
        with gr.Column():
            status_msg = gr.Textbox(label="Status", interactive=False)
            output_md = gr.Code(label="Extracted Simple Markdown", language="markdown")
    
    with gr.Row():
        output_pdf = gr.File(label="Annotated PDF Layout")
        output_gallery = gr.Gallery(label="Extracted Images (Figures/Tables)")

    process_btn.click(
        fn=gradio_process,
        inputs=[input_pdf, mode_input],
        outputs=[output_md, output_gallery, output_pdf, status_msg]
    )


# --------------------------------------------------------------------------------
# Integrate Gradio with FastAPI
# --------------------------------------------------------------------------------
# Mount Gradio at /gradio path (this ensures static files work correctly)
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Redirect root to Gradio interface
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_redirect():
    """Redirect to Gradio interface."""
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/gradio/" />')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

