# MinerU Framework - Comprehensive Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Processing Pipeline](#processing-pipeline)
6. [Model Architectures](#model-architectures)
7. [Data Flow](#data-flow)
8. [Backend Systems](#backend-systems)
9. [API & Integration](#api--integration)
10. [Configuration & Setup](#configuration--setup)

---

## Overview

### What is MinerU?
MinerU is a **high-performance PDF/document parsing framework** that converts PDF documents to Markdown format with exceptional accuracy. It leverages state-of-the-art AI models to extract and structure content including text, tables, formulas, images, and layout information.

### Key Capabilities
- **Document Parsing**: PDF and image files to Markdown conversion
- **Layout Detection**: Identifies titles, paragraphs, tables, images, equations
- **OCR**: Multi-language optical character recognition (15+ languages)
- **Formula Recognition**: Mathematical equation extraction (inline & block)
- **Table Parsing**: Complex table structure recognition (wired/wireless)
- **Multi-Backend**: Pipeline (traditional ML) and VLM (Vision-Language Model) backends
- **Batch Processing**: GPU-accelerated inference with dynamic batching

### Version Information
- **Current Version**: 2.6.3 (as of October 2025)
- **Python Support**: 3.10, 3.11, 3.12, 3.13
- **License**: AGPL-3.0

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   CLI    │  │ FastAPI  │  │  Gradio  │  │   MCP    │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Selection Layer                     │
│  ┌────────────────────┐           ┌─────────────────────────┐   │
│  │  Pipeline Backend  │           │     VLM Backend         │   │
│  │  (Traditional ML)  │           │  (MinerU2.5 Model)      │   │
│  └────────────────────┘           └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Processing Pipeline                        │
│  PDF Input → Layout → OCR → Formula → Table → Markdown Output   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Model Layer                              │
│  Layout │ MFD │ MFR │ OCR │ Table │ Orientation │ Reading Order │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Principles
1. **Modular Design**: Separate models for different tasks (layout, OCR, formula, table)
2. **Singleton Pattern**: Model instances cached for performance
3. **Batch Processing**: Dynamic batching based on GPU memory
4. **Multi-Backend**: Supports both pipeline and VLM approaches
5. **Async Support**: Asynchronous processing for VLM backends

---

## Directory Structure

```
MinerU/
├── mineru/                          # Core package
│   ├── backend/                     # Backend implementations
│   │   ├── pipeline/               # Traditional ML pipeline
│   │   │   ├── model_init.py      # Model initialization
│   │   │   ├── batch_analyze.py   # Batch processing logic
│   │   │   ├── pipeline_analyze.py # Main analysis orchestrator
│   │   │   ├── model_json_to_middle_json.py # Result transformation
│   │   │   ├── pipeline_middle_json_mkcontent.py # Markdown generation
│   │   │   └── model_list.py      # Atomic model definitions
│   │   ├── vlm/                    # Vision-Language Model backend
│   │   │   ├── vlm_analyze.py     # VLM analysis
│   │   │   ├── vlm_magic_model.py # VLM model wrapper
│   │   │   └── vlm_middle_json_mkcontent.py # VLM markdown gen
│   │   └── utils.py                # Backend utilities
│   │
│   ├── cli/                         # Command-line interfaces
│   │   ├── client.py               # Main CLI entry point
│   │   ├── fast_api.py             # FastAPI server
│   │   ├── gradio_app.py           # Gradio web interface
│   │   ├── vlm_vllm_server.py      # vLLM server for VLM
│   │   ├── models_download.py      # Model download utility
│   │   └── common.py               # Shared CLI functions
│   │
│   ├── model/                       # AI Models
│   │   ├── layout/                 # Layout detection
│   │   │   └── doclayoutyolo.py   # DocLayoutYOLO model
│   │   ├── mfd/                    # Math Formula Detection
│   │   │   └── yolo_v8.py         # YOLOv8 MFD model
│   │   ├── mfr/                    # Math Formula Recognition
│   │   │   ├── unimernet/         # UnimerNet model (default)
│   │   │   └── pp_formulanet_plus_m/ # PaddleFormula (Chinese support)
│   │   ├── ocr/                    # Optical Character Recognition
│   │   │   └── pytorch_paddle.py  # PytorchPaddle OCR
│   │   ├── table/                  # Table recognition
│   │   │   ├── rec/               # Table structure recognition
│   │   │   │   ├── slanet_plus/   # Wireless table model
│   │   │   │   └── unet_table/    # Wired table model
│   │   │   └── cls/               # Table classification
│   │   ├── ori_cls/                # Orientation classification
│   │   ├── reading_order/          # Reading order detection
│   │   │   ├── layout_reader.py   # Layout reading order
│   │   │   └── xycut.py          # XY-Cut algorithm
│   │   └── vlm_vllm_model/         # VLM inference server
│   │
│   ├── data/                        # Data I/O
│   │   ├── data_reader_writer/    # File system operations
│   │   └── io/                     # Input/Output utilities
│   │
│   ├── utils/                       # Utility modules
│   │   ├── config_reader.py       # Configuration management
│   │   ├── pdf_reader.py          # PDF parsing
│   │   ├── pdf_image_tools.py     # PDF to image conversion
│   │   ├── ocr_utils.py           # OCR helpers
│   │   ├── model_utils.py         # Model utilities
│   │   ├── enum_class.py          # Enumerations & constants
│   │   ├── draw_bbox.py           # Visualization
│   │   └── [50+ utility files]
│   │
│   └── resources/                   # Static resources
│
├── projects/                        # Extended projects
│   ├── mineru_tianshu/             # TianShu distributed system
│   │   ├── api_server.py          # API server
│   │   ├── task_scheduler.py      # Task scheduling
│   │   └── litserve_worker.py     # Worker processes
│   ├── multi_gpu_v2/               # Multi-GPU inference
│   │   ├── server.py              # Multi-GPU server
│   │   └── client.py              # Client interface
│   └── mcp/                        # Model Context Protocol
│
├── demo/                            # Example files
│   └── pdfs/                       # Sample PDFs
├── docs/                            # Documentation
├── tests/                           # Unit tests
├── docker/                          # Docker configurations
├── pyproject.toml                   # Project configuration
└── README.md                        # Documentation

```

---

## Core Components

### 1. Entry Points (mineru/cli/)

#### **client.py** - Main CLI Interface
**Location**: `mineru/cli/client.py:153-223`

**Responsibilities**:
- Parse command-line arguments
- Configure backend (pipeline/vlm)
- Set device mode (CPU/CUDA/NPU/MPS)
- Orchestrate parsing workflow

**Key Functions**:
```python
def main(ctx, input_path, output_dir, method, backend, lang,
         server_url, start_page_id, end_page_id,
         formula_enable, table_enable, device_mode, virtual_vram, model_source)
```

**Command Options**:
- `-p/--path`: Input file/directory
- `-o/--output`: Output directory
- `-m/--method`: Parse method (auto/txt/ocr)
- `-b/--backend`: Backend selection (pipeline/vlm-*)
- `-l/--lang`: Language for OCR
- `-f/--formula`: Enable formula parsing
- `-t/--table`: Enable table parsing
- `-d/--device`: Device mode

**Example Usage**:
```bash
mineru -p document.pdf -o ./output -b pipeline -l en
```

#### **fast_api.py** - REST API Server
**Location**: `mineru/cli/fast_api.py:1-80`

**Endpoints**:
- `POST /file_parse`: Parse uploaded files
  - Accepts multiple files
  - Supports all backend configurations
  - Returns JSON/Markdown/ZIP results

**Parameters**:
- `files`: Uploaded PDF/image files
- `backend`: Backend selection
- `lang_list`: Languages for OCR
- `return_md`: Return markdown
- `return_middle_json`: Return intermediate JSON
- `return_content_list`: Return content list JSON
- `return_images`: Include extracted images
- `response_format_zip`: Return as ZIP archive

#### **gradio_app.py** - Web Interface
**Location**: `mineru/cli/gradio_app.py`

**Features**:
- Drag-and-drop file upload
- Real-time processing status
- Visual preview of results
- Configuration panel

#### **models_download.py** - Model Management
**Location**: `mineru/cli/models_download.py`

**Functionality**:
- Download models from HuggingFace/ModelScope
- Verify model integrity
- Cache management

### 2. Backend Layer (mineru/backend/)

#### **A. Pipeline Backend** (`mineru/backend/pipeline/`)

##### **pipeline_analyze.py** - Document Analysis Orchestrator
**Location**: `mineru/backend/pipeline/pipeline_analyze.py:70-206`

**Core Function**: `doc_analyze()`
```python
def doc_analyze(pdf_bytes_list, lang_list, parse_method='auto',
                formula_enable=True, table_enable=True)
```

**Process Flow**:
1. **PDF Classification**: Determine if OCR is needed
   - `classify(pdf_bytes)` → 'txt' or 'ocr'

2. **Image Loading**: Convert PDF pages to images
   - `load_images_from_pdf()` → PIL images

3. **Batch Preparation**: Group pages for processing
   - Min batch size: 384 (configurable via `MINERU_MIN_BATCH_INFERENCE_SIZE`)

4. **Batch Analysis**: Process in batches
   - `batch_image_analyze()` → layout detections

5. **Result Assembly**: Organize per-page results

**Batch Size Calculation** (`mineru/backend/pipeline/pipeline_analyze.py:173-191`):
```python
if gpu_memory >= 16: batch_ratio = 16
elif gpu_memory >= 12: batch_ratio = 8
elif gpu_memory >= 8: batch_ratio = 4
elif gpu_memory >= 6: batch_ratio = 2
else: batch_ratio = 1
```

##### **batch_analyze.py** - Batch Processing Engine
**Location**: `mineru/backend/pipeline/batch_analyze.py:25-428`

**Class**: `BatchAnalyze`

**Processing Stages**:

1. **Layout Detection** (Lines 52-54):
   ```python
   images_layout_res = model.layout_model.batch_predict(
       pil_images, YOLO_LAYOUT_BASE_BATCH_SIZE
   )
   ```
   - Detects: text, title, images, tables, equations
   - Model: DocLayoutYOLO

2. **Formula Detection & Recognition** (Lines 56-71):
   ```python
   # Detection
   images_mfd_res = model.mfd_model.batch_predict(np_images, MFD_BASE_BATCH_SIZE)

   # Recognition
   images_formula_list = model.mfr_model.batch_predict(
       images_mfd_res, np_images, batch_size=batch_ratio * MFR_BASE_BATCH_SIZE
   )
   ```
   - MFD: YOLOv8 for formula detection
   - MFR: UnimerNet or PP-FormulaNet for recognition

3. **Table Processing** (Lines 112-226):
   - **Orientation Classification**: Rotate tables if needed
   - **Table Type Classification**: Wired vs wireless
   - **OCR Detection**: Detect text boxes in tables
   - **OCR Recognition**: Recognize text in boxes
   - **Structure Recognition**: Parse table structure

4. **OCR Detection** (Lines 238-364):
   - Two modes: batch and single
   - **Batch Mode**: Groups by language and resolution
   - **Single Mode**: Process individually

5. **OCR Recognition** (Lines 366-427):
   - Group by language
   - Batch recognition per language
   - Confidence filtering

**Batch Size Constants**:
```python
YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16
TABLE_ORI_CLS_BATCH_SIZE = 16
TABLE_Wired_Wireless_CLS_BATCH_SIZE = 16
```

##### **model_init.py** - Model Initialization
**Location**: `mineru/backend/pipeline/model_init.py:1-270`

**Singleton Pattern**: `AtomModelSingleton` (Lines 120-151)
- Caches model instances
- Reuses models with same configuration
- Key: (model_name, lang, other_params)

**Class**: `MineruPipelineModel` (Lines 200-270)

**Models Initialized**:

1. **Layout Model** (Lines 238-245):
   ```python
   self.layout_model = DocLayoutYOLOModel(doclayout_yolo_weights, device)
   ```
   - Path: `models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt`

2. **MFD Model** (Lines 214-221):
   ```python
   self.mfd_model = YOLOv8MFDModel(mfd_weights, device)
   ```
   - Path: `models/MFD/YOLO/yolo_v8_ft.pt`

3. **MFR Model** (Lines 224-236):
   ```python
   self.mfr_model = UnimernetModel(mfr_weight_dir, device)
   # OR
   self.mfr_model = FormulaRecognizer(mfr_weight_dir, device)
   ```
   - UnimerNet (default): `models/MFR/unimernet_hf_small_2503`
   - PP-FormulaNet (Chinese): `models/MFR/pp_formulanet_plus_m`
   - Controlled by: `MINERU_FORMULA_CH_SUPPORT` env variable

4. **OCR Model** (Lines 247-251):
   ```python
   self.ocr_model = PytorchPaddleOCR(lang=lang)
   ```
   - Path: `models/OCR/paddleocr_torch`
   - Supports 15+ languages

5. **Table Models** (Lines 253-268):
   ```python
   self.wired_table_model = UnetTableModel(ocr_engine)
   self.wireless_table_model = RapidTableModel(ocr_engine)
   self.table_cls_model = PaddleTableClsModel()
   self.img_orientation_cls_model = PaddleOrientationClsModel()
   ```

##### **model_json_to_middle_json.py** - Result Transformation
**Location**: `mineru/backend/pipeline/model_json_to_middle_json.py:1-100`

**Function**: `result_to_middle_json()`

**Transforms**:
- Model output → Intermediate JSON format
- Processes: spans, blocks, images, tables
- Applies: span preprocessing, block sorting, content extraction

**Key Operations** (Lines 28-100):
1. Page-by-page processing
2. Span extraction from layout
3. Block boundary preparation
4. Overlap removal
5. Block-span filling
6. Image/table cutting
7. Paragraph splitting

##### **pipeline_middle_json_mkcontent.py** - Markdown Generation
**Function**: `union_make()`

**Generates**:
- Markdown text from intermediate JSON
- Content list JSON
- Handles: text, images, tables, equations
- Preserves: reading order, formatting

#### **B. VLM Backend** (`mineru/backend/vlm/`)

##### **vlm_analyze.py** - VLM Analysis
**Location**: `mineru/backend/vlm/vlm_analyze.py`

**Function**: `doc_analyze()` and `aio_doc_analyze()`

**VLM Backends**:
1. **transformers**: HuggingFace Transformers
2. **mlx-engine**: Apple MLX (Mac only)
3. **vllm-engine**: vLLM inference engine
4. **vllm-async-engine**: Async vLLM
5. **http-client**: HTTP API client

**Process**:
1. Convert PDF to images
2. Send images to VLM model (MinerU2.5-1.2B)
3. Model outputs structured JSON
4. Transform to middle JSON format

##### **vlm_magic_model.py** - VLM Model Wrapper
**Location**: `mineru/backend/vlm/vlm_magic_model.py`

**Responsibilities**:
- Load MinerU2.5 model
- Inference orchestration
- Output parsing

**Model**: MinerU2.5-2509-1.2B
- Parameters: 1.2B
- Architecture: Vision-Language Transformer
- Performance: SOTA on OmniDocBench

### 3. Model Layer (mineru/model/)

#### **Layout Detection** (`mineru/model/layout/`)

##### **DocLayoutYOLO**
**File**: `mineru/model/layout/doclayoutyolo.py`

**Model Type**: YOLO-based object detection

**Detectable Categories** (from `enum_class.py:41-56`):
- 0: Title
- 1: Text
- 2: Abandon (discarded content)
- 3: ImageBody
- 4: ImageCaption
- 5: TableBody
- 6: TableCaption
- 7: TableFootnote
- 8: InterlineEquation_Layout
- 9: InterlineEquationNumber_Layout
- 13: InlineEquation
- 14: InterlineEquation_YOLO
- 15: OcrText
- 16: LowScoreText
- 101: ImageFootnote

**Output**: Bounding boxes with category IDs

#### **Math Formula Detection (MFD)** (`mineru/model/mfd/`)

##### **YOLOv8MFDModel**
**File**: `mineru/model/mfd/yolo_v8.py`

**Purpose**: Detect mathematical formulas in images

**Process**:
1. Detect formula regions (bounding boxes)
2. Return coordinates for MFR processing

**Types Detected**:
- Inline equations
- Block equations
- Equation numbers

#### **Math Formula Recognition (MFR)** (`mineru/model/mfr/`)

##### **A. UnimerNet (Default)**
**Directory**: `mineru/model/mfr/unimernet/`

**Model**: UnimerNet Small
**Strengths**:
- Fast inference
- English formulas
- LaTeX output

##### **B. PP-FormulaNet Plus M**
**Directory**: `mineru/model/mfr/pp_formulanet_plus_m/`

**Model**: PaddlePaddle FormulaNet
**Strengths**:
- Chinese formula support
- Mixed Chinese-English formulas

**Activation**:
```bash
export MINERU_FORMULA_CH_SUPPORT=1
```

**Output**: LaTeX code

#### **OCR** (`mineru/model/ocr/`)

##### **PytorchPaddleOCR**
**File**: `mineru/model/ocr/pytorch_paddle.py`

**Based On**: PaddleOCR (PyTorch port)

**Components**:
1. **Text Detector**: Detects text regions
   - DB (Differentiable Binarization)
   - Outputs: quadrilateral boxes

2. **Text Recognizer**: Recognizes text
   - CRNN-based
   - Outputs: text + confidence

**Supported Languages** (from CLI):
- ch (Chinese)
- ch_server (Chinese server)
- ch_lite (Chinese lite)
- en (English)
- korean
- japan
- chinese_cht (Traditional Chinese)
- ta (Tamil)
- te (Telugu)
- ka (Kannada)
- th (Thai)
- el (Greek)
- latin
- arabic
- east_slavic
- cyrillic
- devanagari

**Parameters**:
- `det_db_box_thresh`: Detection threshold (default: 0.3)
- `det_db_unclip_ratio`: Unclip ratio (default: 1.8)
- `enable_merge_det_boxes`: Merge nearby boxes

#### **Table Recognition** (`mineru/model/table/`)

##### **A. Table Classification**
**File**: `mineru/model/table/cls/paddle_table_cls.py`

**Purpose**: Classify table type
- **Wired**: Tables with visible borders
- **Wireless**: Borderless tables

**Model**: PP-LCNet

##### **B. Wired Table Model**
**Directory**: `mineru/model/table/rec/unet_table/`

**Model**: U-Net based structure recognition

**Process**:
1. Detect table lines
2. Find cell boundaries
3. Extract structure
4. Generate HTML

##### **C. Wireless Table Model**
**Directory**: `mineru/model/table/rec/slanet_plus/`

**Model**: SLANet+ (Structure Layout Analysis Network)

**Process**:
1. Analyze spatial relationships
2. Infer cell structure
3. Generate HTML

**Output**: HTML table code

#### **Orientation Classification** (`mineru/model/ori_cls/`)

##### **PaddleOrientationClsModel**
**File**: `mineru/model/ori_cls/paddle_ori_cls.py`

**Purpose**: Detect image/table rotation
**Angles**: 0°, 90°, 180°, 270°

#### **Reading Order** (`mineru/model/reading_order/`)

##### **Layout Reader**
**File**: `mineru/model/reading_order/layout_reader.py`

**Purpose**: Determine reading order of blocks

##### **XY-Cut Algorithm**
**File**: `mineru/model/reading_order/xycut.py`

**Algorithm**: Recursive XY projection cutting
**Use Case**: Multi-column layouts

---

## Processing Pipeline

### Pipeline Backend Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PDF INPUT PROCESSING                                         │
│    - Read PDF bytes                                             │
│    - Classify: txt vs ocr (if parse_method='auto')             │
│    - Convert pages to images (PIL)                              │
│    - Page range selection (start_page_id, end_page_id)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. BATCH PREPARATION                                            │
│    - Group pages into batches                                   │
│    - Batch size: MINERU_MIN_BATCH_INFERENCE_SIZE (default: 384)│
│    - Prepare: (image, ocr_enable, lang) tuples                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. LAYOUT DETECTION (DocLayoutYOLO)                            │
│    - Input: PIL images                                          │
│    - Process: batch_predict(images, batch_size=1)              │
│    - Output: Bounding boxes + category IDs                     │
│    - Categories: text, title, image, table, equation, etc.     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. FORMULA PROCESSING (if formula_enable=True)                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 4a. Formula Detection (YOLOv8 MFD)                 │     │
│    │     - Detect formula regions                        │     │
│    │     - Batch size: 1                                 │     │
│    └─────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 4b. Formula Recognition (UnimerNet/PP-FormulaNet)  │     │
│    │     - Extract LaTeX from formula images             │     │
│    │     - Batch size: 16 * batch_ratio                 │     │
│    └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. TABLE PROCESSING (if table_enable=True)                     │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 5a. Orientation Classification                      │     │
│    │     - Detect table rotation (0/90/270°)            │     │
│    │     - Rotate if needed                              │     │
│    └─────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 5b. Table Type Classification                       │     │
│    │     - Classify: wired vs wireless                   │     │
│    └─────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 5c. OCR on Table Cells                              │     │
│    │     - Detection: find text boxes                    │     │
│    │     - Recognition: extract text                     │     │
│    │     - Batch size: 16 * batch_ratio                 │     │
│    └─────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 5d. Structure Recognition                           │     │
│    │     - Wireless: SLANet+ model                       │     │
│    │     - Wired: U-Net model (if confidence < 0.9)     │     │
│    │     - Output: HTML table structure                  │     │
│    └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. OCR PROCESSING                                               │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 6a. Text Region Detection                           │     │
│    │     - Batch mode: Group by lang & resolution        │     │
│    │     - Resolution padding: 64px stride               │     │
│    │     - Batch size: 16 * batch_ratio                 │     │
│    └─────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ 6b. Text Recognition                                │     │
│    │     - Group by language                             │     │
│    │     - Batch recognition per language                │     │
│    │     - Confidence filtering (min: varies)            │     │
│    └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. POST-PROCESSING                                              │
│    - Span preprocessing (overlap removal, sorting)             │
│    - Block sorting by reading order                            │
│    - Fill spans into blocks                                     │
│    - Fix discarded blocks                                       │
│    - Cross-page table merging                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. CONTENT EXTRACTION                                           │
│    - Cut images and tables from pages                          │
│    - Save images to output directory                            │
│    - Generate image references                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. MARKDOWN GENERATION                                          │
│    - Transform middle JSON to Markdown                          │
│    - Apply formatting (headers, lists, tables, equations)      │
│    - Insert image references                                    │
│    - Generate content_list.json (structured output)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. OUTPUT                                                       │
│     - {filename}.md: Markdown output                            │
│     - {filename}_middle.json: Intermediate format               │
│     - {filename}_model.json: Raw model output                   │
│     - {filename}_content_list.json: Structured content          │
│     - images/: Extracted images and tables                      │
│     - {filename}_layout.pdf: Layout visualization (optional)    │
└─────────────────────────────────────────────────────────────────┘
```

### VLM Backend Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PDF INPUT                                                     │
│    - Read PDF bytes                                             │
│    - Convert pages to images                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. VLM INFERENCE                                                 │
│    - Load MinerU2.5 model (1.2B params)                         │
│    - Send images to model                                        │
│    - Model outputs structured JSON                              │
│    - Includes: layout, text, tables, formulas, reading order   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. POST-PROCESSING                                              │
│    - Parse VLM JSON output                                      │
│    - Extract images and tables                                  │
│    - Cross-page table merging                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. MARKDOWN GENERATION                                          │
│    - Convert to middle JSON format                              │
│    - Generate Markdown                                           │
│    - Create content_list.json                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Input Formats
1. **PDF Files**: Native PDF documents
2. **Images**: PNG, JPEG, JP2, WebP, GIF, BMP, TIFF

### Internal Data Structures

#### **1. Layout Detection Output**
```json
{
  "category_id": 1,          // Block type (see CategoryId)
  "poly": [x1, y1, x2, y2, x3, y3, x4, y4],  // Bounding box
  "score": 0.95,             // Confidence score
  "text": "content",         // OCR text (if applicable)
}
```

#### **2. Middle JSON Format**
```json
{
  "pdf_info": [
    {
      "page_no": 0,
      "page_size": {"width": 595, "height": 842},
      "blocks": [
        {
          "type": "text",
          "bbox": [x1, y1, x2, y2],
          "content": "text content",
          "spans": [...]
        },
        {
          "type": "table",
          "bbox": [x1, y1, x2, y2],
          "html": "<table>...</table>",
          "image_path": "images/table_0.png"
        },
        {
          "type": "image",
          "bbox": [x1, y1, x2, y2],
          "image_path": "images/image_0.png"
        },
        {
          "type": "equation",
          "bbox": [x1, y1, x2, y2],
          "latex": "E = mc^2"
        }
      ]
    }
  ]
}
```

#### **3. Content List JSON** (User-facing)
```json
[
  {
    "type": "text",
    "text": "paragraph content",
    "bbox": [x1, y1, x2, y2]  // Normalized to 0-1000
  },
  {
    "type": "table",
    "html": "<table>...</table>",
    "img_path": "images/table_0.png"
  },
  {
    "type": "image",
    "img_path": "images/image_0.png"
  }
]
```

### Output Files

For input `document.pdf`, output directory structure:
```
output/
└── document/
    └── auto/  (or vlm/)
        ├── document.md                    # Markdown output
        ├── document_middle.json           # Intermediate format
        ├── document_model.json            # Raw model output
        ├── document_content_list.json     # Structured content
        ├── document_layout.pdf            # Layout visualization (optional)
        ├── document_span.pdf              # Span visualization (optional)
        ├── document_origin.pdf            # Original PDF (optional)
        └── images/                        # Extracted assets
            ├── image_0.png
            ├── table_0.png
            └── ...
```

---

## Backend Systems

### 1. Pipeline Backend

**Advantages**:
- More control over individual components
- Better for specific use cases
- Faster for documents without complex layouts
- Lower GPU memory requirements

**Disadvantages**:
- Multiple model inference steps
- More complex pipeline
- May miss some layout relationships

**Recommended For**:
- Academic papers
- Technical documentation
- Documents with clear structure

### 2. VLM Backend

**Model**: MinerU2.5-2509-1.2B

**Backends**:

#### **A. transformers** (vlm-transformers)
- **Framework**: HuggingFace Transformers
- **Use Case**: General purpose, CPU/GPU
- **Speed**: Slowest
- **Memory**: Moderate

#### **B. mlx-engine** (vlm-mlx-engine)
- **Framework**: Apple MLX
- **Platform**: Mac with Apple Silicon only
- **Speed**: 100-200% faster than transformers
- **Memory**: Efficient

#### **C. vllm-engine** (vlm-vllm-engine)
- **Framework**: vLLM
- **Use Case**: Single document, high throughput
- **Speed**: Fastest
- **Memory**: High (continuous batching)

#### **D. vllm-async-engine** (vlm-vllm-async-engine)
- **Framework**: vLLM async
- **Use Case**: Multiple documents concurrently
- **Speed**: Fastest for batches
- **Memory**: High

#### **E. http-client** (vlm-http-client)
- **Framework**: HTTP API
- **Use Case**: Remote inference
- **Requires**: Server URL (`-u` parameter)

**Advantages**:
- SOTA accuracy (beats GPT-4o, Gemini 2.5 Pro)
- Single model inference
- Better layout understanding
- Handles complex documents

**Disadvantages**:
- Higher GPU memory (for local inference)
- Requires 1.2B param model download
- Less granular control

**Recommended For**:
- Complex layouts
- Mixed content types
- High accuracy requirements

### Comparison Table

| Feature | Pipeline | VLM (transformers) | VLM (vllm) |
|---------|----------|-------------------|-----------|
| **Accuracy** | Good | Excellent (SOTA) | Excellent (SOTA) |
| **Speed** | Medium | Slow | Fast |
| **GPU Memory** | 4-6GB | 8-16GB | 8-16GB |
| **CPU Support** | Yes | Yes | Limited |
| **Fine Control** | High | Low | Low |
| **Setup** | Complex | Simple | Medium |

---

## API & Integration

### 1. Command-Line Interface

**Entry Point**: `mineru` command

**Installation**:
```bash
pip install mineru[core]  # Full installation
```

**Basic Usage**:
```bash
# Pipeline backend
mineru -p input.pdf -o ./output -b pipeline

# VLM backend with vLLM
mineru -p input.pdf -o ./output -b vlm-vllm-engine

# Specific page range
mineru -p input.pdf -o ./output -s 0 -e 10

# Disable formula/table parsing
mineru -p input.pdf -o ./output -f False -t False

# Multi-language OCR
mineru -p input.pdf -o ./output -l en
```

### 2. REST API

**Start Server**:
```bash
mineru-api
# Server runs on http://localhost:8000
```

**API Endpoint**: `POST /file_parse`

**Example Request** (Python):
```python
import requests

files = [('files', open('document.pdf', 'rb'))]
data = {
    'backend': 'pipeline',
    'lang_list': ['en'],
    'formula_enable': True,
    'table_enable': True,
    'return_md': True,
    'return_content_list': True
}

response = requests.post(
    'http://localhost:8000/file_parse',
    files=files,
    data=data
)

result = response.json()
```

**Example Request** (cURL):
```bash
curl -X POST "http://localhost:8000/file_parse" \
  -F "files=@document.pdf" \
  -F "backend=pipeline" \
  -F "lang_list=en" \
  -F "return_md=true"
```

### 3. Gradio Web Interface

**Start Interface**:
```bash
mineru-gradio
# Opens web interface in browser
```

**Features**:
- File upload
- Configuration options
- Real-time processing
- Preview results
- Download outputs

### 4. Python API

**Synchronous**:
```python
from mineru.cli.common import do_parse, read_fn
from pathlib import Path

# Read PDF
pdf_bytes = read_fn(Path('input.pdf'))

# Parse
do_parse(
    output_dir='./output',
    pdf_file_names=['document'],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=['en'],
    backend='pipeline',
    parse_method='auto',
    formula_enable=True,
    table_enable=True
)
```

**Asynchronous**:
```python
from mineru.cli.common import aio_do_parse, read_fn
from pathlib import Path
import asyncio

async def parse_pdf():
    pdf_bytes = read_fn(Path('input.pdf'))

    await aio_do_parse(
        output_dir='./output',
        pdf_file_names=['document'],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=['en'],
        backend='vlm-vllm-async-engine',
        formula_enable=True,
        table_enable=True
    )

asyncio.run(parse_pdf())
```

### 5. Model Context Protocol (MCP)

**Location**: `projects/mcp/`

**Purpose**: Integration with AI assistants

**Server**: Provides MCP-compatible interface for document parsing

### 6. Multi-GPU Support

**Location**: `projects/multi_gpu_v2/`

**Server Mode**:
```bash
python projects/multi_gpu_v2/server.py
```

**Client Mode**:
```python
from projects.multi_gpu_v2.client import parse_documents
results = parse_documents(pdf_list, server_url='http://localhost:8080')
```

### 7. Distributed Processing (TianShu)

**Location**: `projects/mineru_tianshu/`

**Components**:
- **API Server**: `api_server.py`
- **Task Scheduler**: `task_scheduler.py`
- **Workers**: `litserve_worker.py`

**Architecture**:
- Task queue management
- Worker pool
- Load balancing
- Result aggregation

---

## Configuration & Setup

### Environment Variables

```bash
# Device selection
MINERU_DEVICE_MODE=cuda:0  # cuda, cpu, npu, mps

# GPU memory (GB)
MINERU_VIRTUAL_VRAM_SIZE=16

# Model source
MINERU_MODEL_SOURCE=huggingface  # or modelscope, local

# Batch size
MINERU_MIN_BATCH_INFERENCE_SIZE=384

# Formula model
MINERU_FORMULA_CH_SUPPORT=0  # 0: UnimerNet, 1: PP-FormulaNet

# VLM settings
MINERU_VLM_FORMULA_ENABLE=True
MINERU_VLM_TABLE_ENABLE=True

# Table merging
MINERU_TABLE_MERGE_ENABLE=1  # 0: disable, 1: enable
```

### Configuration File

**Template**: `mineru.template.json`

```json
{
  "device_mode": "cuda",
  "model_source": "huggingface",
  "formula_enable": true,
  "table_enable": true,
  "lang": "en"
}
```

### Model Downloads

**Download All Models**:
```bash
mineru-models-download
```

**Manual Download**:
- **HuggingFace**: `opendatalab/PDF-Extract-Kit-1.0`
- **ModelScope**: `OpenDataLab/PDF-Extract-Kit-1.0`
- **VLM Model**: `opendatalab/MinerU2.5-2509-1.2B`

**Model Paths** (from `enum_class.py:65-79`):
```
models/
├── Layout/
│   └── YOLO/
│       └── doclayout_yolo_docstructbench_imgsz1280_2501.pt
├── MFD/
│   └── YOLO/
│       └── yolo_v8_ft.pt
├── MFR/
│   ├── unimernet_hf_small_2503/
│   └── pp_formulanet_plus_m/
├── OCR/
│   └── paddleocr_torch/
├── TabRec/
│   ├── SlanetPlus/
│   │   └── slanet-plus.onnx
│   └── UnetStructure/
│       └── unet.onnx
├── TabCls/
│   └── paddle_table_cls/
│       └── PP-LCNet_x1_0_table_cls.onnx
└── OriCls/
    └── paddle_orientation_classification/
        └── PP-LCNet_x1_0_doc_ori.onnx
```

### Dependencies

**Core** (`pyproject.toml:19-43`):
- `pdfminer.six`: PDF parsing
- `pypdfium2`: PDF rendering
- `pillow`: Image processing
- `opencv-python`: Computer vision
- `numpy`: Numerical computing

**Pipeline** (`pyproject.toml:64-78`):
- `torch`: Deep learning
- `torchvision`: Vision models
- `ultralytics`: YOLO models
- `doclayout_yolo`: Layout detection
- `transformers`: Transformer models
- `onnxruntime`: ONNX inference

**VLM** (`pyproject.toml:53-62`):
- `torch`: Deep learning
- `transformers`: HuggingFace
- `accelerate`: Training utilities
- `vllm`: Inference engine (optional)
- `mlx-vlm`: Apple MLX (Mac only)

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- GPU: Not required (CPU mode available)

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- GPU: 8GB+ VRAM (NVIDIA/AMD/Apple Silicon)

**Optimal**:
- CPU: 16+ cores
- RAM: 32GB
- GPU: 16GB+ VRAM
- Storage: SSD

### Docker Support

**Directories**: `docker/china/`, `docker/global/`

**Pull Image**:
```bash
docker pull opendatalab/mineru:latest
```

**Run Container**:
```bash
docker run -it --gpus all \
  -v /path/to/pdfs:/data \
  -v /path/to/output:/output \
  opendatalab/mineru:latest \
  mineru -p /data/document.pdf -o /output
```

---

## Advanced Topics

### 1. Batch Size Optimization

**Key Variable**: `batch_ratio` (calculated in `pipeline_analyze.py:173-191`)

**Formula**:
```
effective_batch_size = BASE_BATCH_SIZE * batch_ratio
```

**Base Batch Sizes** (`batch_analyze.py:17-22`):
- Layout: 1
- MFD: 1
- MFR: 16
- OCR Detection: 16
- OCR Recognition: Variable
- Table Classification: 16

**Memory Mapping**:
- 16GB GPU → batch_ratio = 16
- 12GB GPU → batch_ratio = 8
- 8GB GPU → batch_ratio = 4
- 6GB GPU → batch_ratio = 2
- <6GB GPU → batch_ratio = 1

### 2. Cross-Page Table Merging

**Location**: `mineru/backend/utils.py:cross_page_table_merge()`

**Algorithm**:
1. Detect table at page bottom
2. Check for continuation on next page
3. Merge HTML structures
4. Update bounding boxes

**Control**: `MINERU_TABLE_MERGE_ENABLE=1`

### 3. OCR Batch Optimization

**Strategy** (`batch_analyze.py:239-310`):
1. Group images by language
2. Sub-group by resolution (64px stride)
3. Pad to uniform size
4. Batch inference
5. Parse results

**Benefit**: 200-300% speedup

### 4. Formula Recognition Modes

**English (Default)**:
```bash
# Uses UnimerNet Small
mineru -p doc.pdf -o ./output
```

**Chinese Support**:
```bash
export MINERU_FORMULA_CH_SUPPORT=1
mineru -p doc.pdf -o ./output
```

**Trade-offs**:
- Chinese mode: Slower, handles mixed CN/EN
- English mode: Faster, English only

### 5. Table Parsing Algorithm

**Hybrid Approach** (`batch_analyze.py:193-226`):

1. **Classify** all tables (wired/wireless)
2. **Process** with wireless model (SLANet+)
3. **Re-process** low-confidence or wired tables with U-Net
4. **Merge** results

**Threshold**: Confidence < 0.9 triggers wired model

### 6. Reading Order Detection

**Methods**:
1. **Layout Reader**: ML-based ordering
2. **XY-Cut**: Geometric algorithm
3. **Default**: Top-to-bottom, left-to-right

**Selection**: Automatic based on layout complexity

### 7. vLLM Acceleration

**Configuration**:
```bash
# Start vLLM server
mineru-vllm-server

# Use from CLI
mineru -p doc.pdf -o ./output -b vlm-vllm-engine

# Or connect to remote server
mineru -p doc.pdf -o ./output -b vlm-http-client -u http://server:30000
```

**Performance**: 3-10x faster than transformers

### 8. Visualization & Debugging

**Enable Debug Outputs**:
```python
do_parse(
    ...,
    f_draw_layout_bbox=True,    # Visualize layout boxes
    f_draw_span_bbox=True,       # Visualize text spans
    f_dump_model_output=True,    # Save model JSON
    f_dump_middle_json=True      # Save intermediate JSON
)
```

**Outputs**:
- `{file}_layout.pdf`: Layout detection visualization
- `{file}_span.pdf`: Text span visualization
- `{file}_model.json`: Raw model output
- `{file}_middle.json`: Processed intermediate format

---

## Key Algorithms & Techniques

### 1. Layout Detection

**Model**: DocLayoutYOLO
- **Architecture**: YOLOv8-based
- **Input**: 1280x1280 images
- **Output**: Bounding boxes + 17 categories
- **Backbone**: CSPDarknet
- **Neck**: PANet
- **Head**: YOLO detection head

### 2. Formula Recognition

**UnimerNet**:
- **Architecture**: Vision Transformer
- **Input**: Formula image crops
- **Output**: LaTeX string
- **Tokenizer**: BPE
- **Decoder**: Autoregressive

### 3. Table Structure Recognition

**SLANet+**:
- **Architecture**: CNN + GRU
- **Process**: Row/column prediction
- **Output**: Cell coordinates + spans

**U-Net**:
- **Architecture**: U-Net encoder-decoder
- **Process**: Line detection + cell extraction
- **Output**: Cell coordinates

### 4. OCR

**Text Detection (DB)**:
- **Architecture**: ResNet + FPN
- **Process**: Differentiable binarization
- **Output**: Quadrilateral boxes

**Text Recognition (CRNN)**:
- **Architecture**: CNN + BiLSTM + CTC
- **Input**: Text line images
- **Output**: Unicode text + confidence

### 5. Vision-Language Model (MinerU2.5)

**Architecture**:
- **Vision Encoder**: High-res Vision Transformer
- **Language Decoder**: Transformer decoder
- **Parameters**: 1.2B
- **Input**: Full page images
- **Output**: Structured JSON (layout + content)

**Training**:
- Two-stage: Layout detection → Content recognition
- Dataset: OmniDocBench + proprietary data
- Optimization: Mixed precision, gradient checkpointing

---

## Performance Metrics

### Accuracy (OmniDocBench Benchmark)

| Model | Overall Score |
|-------|--------------|
| **MinerU2.5-1.2B** | **78.4** |
| Gemini 2.5 Pro | 74.2 |
| GPT-4o | 71.8 |
| Qwen2.5-VL-72B | 70.5 |
| Pipeline Backend | 65.3 |

### Speed Benchmarks

**Pipeline Backend** (16GB GPU):
- Simple PDF (10 pages): ~30s
- Complex PDF (10 pages): ~60s
- Throughput: ~10-20 pages/min

**VLM Backend** (transformers, 16GB GPU):
- Simple PDF (10 pages): ~120s
- Complex PDF (10 pages): ~180s
- Throughput: ~3-5 pages/min

**VLM Backend** (vLLM, 16GB GPU):
- Simple PDF (10 pages): ~40s
- Complex PDF (10 pages): ~60s
- Throughput: ~10-15 pages/min

**VLM Backend** (mlx-engine, M3 Max):
- Simple PDF (10 pages): ~60s
- Complex PDF (10 pages): ~90s
- Throughput: ~6-10 pages/min

---

## Troubleshooting & Common Issues

### Issue 1: Out of Memory
**Symptom**: GPU OOM during processing
**Solutions**:
- Reduce `MINERU_MIN_BATCH_INFERENCE_SIZE`
- Set `MINERU_VIRTUAL_VRAM_SIZE` lower
- Disable formula/table: `-f False -t False`
- Use CPU mode: `-d cpu`

### Issue 2: Slow Processing
**Symptoms**: Very slow inference
**Solutions**:
- Use GPU: `-d cuda:0`
- Increase batch size: `export MINERU_MIN_BATCH_INFERENCE_SIZE=512`
- Use vLLM backend: `-b vlm-vllm-engine`
- Enable OCR batch mode (enabled by default in torch <2.8)

### Issue 3: Poor OCR Accuracy
**Symptoms**: Incorrect text extraction
**Solutions**:
- Specify correct language: `-l en`
- Force OCR mode: `-m ocr`
- Check image quality
- Adjust OCR thresholds in config

### Issue 4: Table Not Recognized
**Symptoms**: Tables parsed as text
**Solutions**:
- Ensure `table_enable=True`
- Use VLM backend for complex tables
- Check table visibility (contrast)
- Enable wired table model

### Issue 5: Formula Recognition Errors
**Symptoms**: Wrong LaTeX output
**Solutions**:
- Use Chinese formula model: `export MINERU_FORMULA_CH_SUPPORT=1`
- Check formula clarity
- Increase image resolution
- Use VLM backend

---

## Future Roadmap & Extensions

### Planned Features
1. Multi-modal output (DOCX, HTML, JSON-LD)
2. Streaming API for large documents
3. Fine-tuning API for custom domains
4. Cloud-native deployment (K8s)
5. Real-time collaboration features

### Extensibility Points

**Custom Models**:
- Replace layout model: Implement `DocLayoutYOLOModel` interface
- Custom OCR: Implement `PytorchPaddleOCR` interface
- Custom table parser: Implement table model interface

**Custom Backends**:
- Add new backend in `mineru/backend/`
- Implement `doc_analyze()` function
- Register in `common.py:do_parse()`

**Custom Output Formats**:
- Extend `pipeline_middle_json_mkcontent.py`
- Implement new `make_*()` function
- Add to `MakeMode` enum

---

## References

### Documentation
- **Official Docs**: https://opendatalab.github.io/MinerU/
- **GitHub**: https://github.com/opendatalab/MinerU
- **Technical Report**: https://arxiv.org/abs/2509.22186

### Model Repositories
- **HuggingFace**: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
- **ModelScope**: https://modelscope.cn/models/OpenDataLab/MinerU2.5-2509-1.2B

### Related Projects
- **PDF-Extract-Kit**: Base model collection
- **DocLayout-YOLO**: Layout detection
- **UnimerNet**: Formula recognition
- **PaddleOCR**: OCR engine
- **vLLM**: Inference optimization

---

## Glossary

- **MFD**: Math Formula Detection
- **MFR**: Math Formula Recognition
- **OCR**: Optical Character Recognition
- **VLM**: Vision-Language Model
- **YOLO**: You Only Look Once (object detection)
- **CRNN**: Convolutional Recurrent Neural Network
- **DB**: Differentiable Binarization
- **CTC**: Connectionist Temporal Classification
- **vLLM**: Very Large Language Model (inference framework)
- **MLX**: Apple's machine learning framework
- **ONNX**: Open Neural Network Exchange
- **Bbox**: Bounding box
- **Span**: Text segment with position
- **Block**: Document region (text, image, table, etc.)
- **Middle JSON**: Intermediate representation format
- **Content List**: Structured output format
- **Reading Order**: Sequence of content blocks
- **Cross-page Merge**: Combining split tables/figures across pages

---

## Code Entry Points Reference

### CLI Entry Points (pyproject.toml:106-111)
- `mineru`: Main CLI → `mineru.cli.client:main`
- `mineru-vllm-server`: VLM server → `mineru.cli.vlm_vllm_server:main`
- `mineru-models-download`: Download models → `mineru.cli.models_download:download_models`
- `mineru-api`: FastAPI server → `mineru.cli.fast_api:main`
- `mineru-gradio`: Web UI → `mineru.cli.gradio_app:main`

### Main Processing Functions
- **Pipeline**: `mineru/backend/pipeline/pipeline_analyze.py:doc_analyze()`
- **VLM**: `mineru/backend/vlm/vlm_analyze.py:doc_analyze()`
- **Batch Processing**: `mineru/backend/pipeline/batch_analyze.py:BatchAnalyze.__call__()`
- **Markdown Gen**: `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py:union_make()`

### Model Classes
- **Layout**: `mineru/model/layout/doclayoutyolo.py:DocLayoutYOLOModel`
- **MFD**: `mineru/model/mfd/yolo_v8.py:YOLOv8MFDModel`
- **MFR**: `mineru/model/mfr/unimernet/Unimernet.py:UnimernetModel`
- **OCR**: `mineru/model/ocr/pytorch_paddle.py:PytorchPaddleOCR`
- **Table**: `mineru/model/table/rec/slanet_plus/main.py:RapidTableModel`

---

**Document Version**: 1.0
**Last Updated**: November 3, 2025
**MinerU Version**: 2.6.3
**Author**: Architecture Analysis Tool
