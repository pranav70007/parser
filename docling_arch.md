# Docling Framework Architecture

## Executive Summary

Docling is a sophisticated document understanding framework built with a modular, pipeline-based architecture. It processes various document formats (PDF, DOCX, HTML, images, audio, etc.) through specialized pipelines that orchestrate multiple AI models for layout analysis, OCR, table structure extraction, and enrichment tasks. The framework outputs structured `DoclingDocument` objects that can be used for RAG, search, and other downstream applications.

**Key Capabilities:**
- Multi-format document processing (20+ formats)
- Advanced PDF understanding with AI models
- Vision-Language Model (VLM) integration
- Structured data extraction
- Audio/video transcription
- Multi-threaded, production-ready pipelines
- Extensible plugin architecture

---

## Table of Contents

1. [Core Pipeline Architecture](#1-core-pipeline-architecture)
2. [Data Models](#2-data-models)
3. [Backend & Converters](#3-backend--converters)
4. [Models & Processing](#4-models--processing)
5. [Chunking & Post-processing](#5-chunking--post-processing)
6. [Document Processing Flow](#6-document-processing-flow)
7. [Configuration & Customization](#7-configuration--customization)
8. [Key Design Patterns](#8-key-design-patterns)
9. [Dependencies & Relationships](#9-dependencies--relationships)
10. [Output Formats](#10-output-formats)

---

## 1. Core Pipeline Architecture

### 1.1 Base Pipeline Hierarchy

**Location:** `docling/pipeline/base_pipeline.py`

#### BasePipeline (Abstract Base Class)

The foundation for all pipelines, defining the core conversion workflow.

**Key Responsibilities:**
- Document building (`_build_document`)
- Document assembly (`_assemble_document`)
- Document enrichment (`_enrich_document`)
- Status determination
- Resource cleanup

**Key Methods:**
```python
def execute(self, in_doc: InputDocument) -> ConversionResult:
    """Main entry point orchestrating the 3-stage process"""
    # 1. Build: Extract structure and predictions
    # 2. Assemble: Create document hierarchy
    # 3. Enrich: Apply post-processing models
```

**Abstract Methods (must be implemented by subclasses):**
- `_build_document()`: Core conversion logic
- `_determine_status()`: Success/failure determination

**Configuration:**
- `pipeline_options`: Pipeline-specific options
- `artifacts_path`: Location of model artifacts
- `build_pipe`: List of build stage callables
- `enrichment_pipe`: List of enrichment models

#### ConvertPipeline

**Extends:** `BasePipeline`

**Purpose:** Base for pipelines that convert documents with enrichment

**Key Features:**
- Picture description model integration
- Picture classification support
- Common enrichment models for all backends

#### PaginatedPipeline

**Extends:** `ConvertPipeline`

**Purpose:** Handles paginated documents (PDFs, images)

**Key Features:**
- Page batching (configurable batch size)
- Document timeout handling
- Page initialization and backend loading
- Per-page processing with pipeline stages
- Image caching control
- Backend cleanup

**Processing Flow:**
1. Create page placeholders within specified page range
2. Batch pages (configurable size)
3. Initialize page resources
4. Apply pipeline stages to each batch
5. Cleanup images/backends based on configuration
6. Filter out uninitialized pages after timeout/failures

---

### 1.2 Standard PDF Pipeline

**Location:** `docling/pipeline/standard_pdf_pipeline.py`

#### StandardPdfPipeline (ThreadedPdfPipeline)

High-performance, production-ready PDF processing with multi-threaded stages.

**Architecture:** Thread-safe pipeline with per-run isolation and deterministic execution

**Key Design Principles:**
- **Per-run isolation**: Each `execute()` call uses its own queues and worker threads
- **Deterministic run IDs**: Pages tracked with monotonic run-id (not object `id()`)
- **Explicit back-pressure**: Producers block on full queues
- **Minimal shared state**: Models initialized once, read-only by workers
- **Strict typing**: Fully annotated with clean API usage

**Pipeline Stages** (each runs in separate thread):

1. **Preprocess Stage:**
   - Lazily loads PDF page backends
   - Runs `PagePreprocessingModel`
   - Initializes page size

2. **OCR Stage:**
   - Configurable batch size (default: 4)
   - Processes text extraction
   - Handles OCR model execution

3. **Layout Stage:**
   - Runs `LayoutModel` for document structure
   - Configurable batch size (default: 4)
   - Identifies text blocks, tables, figures

4. **Table Stage:**
   - Runs `TableStructureModel`
   - Extracts table structure and cells
   - Configurable batch size (default: 4)

5. **Assemble Stage:**
   - Runs `PageAssembleModel`
   - Creates page elements
   - Releases page resources after processing

**Models Initialized:**
- `PagePreprocessingModel`
- OCR model (via factory)
- `LayoutModel`
- `TableStructureModel`
- `PageAssembleModel`
- `ReadingOrderModel`
- `CodeFormulaModel` (optional enrichment)

**Threading Components:**
- `ThreadedQueue`: Bounded queue with blocking put/get and explicit close semantics
- `ThreadedPipelineStage`: Single stage backed by one worker thread
- `PreprocessThreadedStage`: Specialized stage for lazy backend loading
- `RunContext`: Wiring for a single `execute()` call

**Document Assembly:**
- Aggregates elements/headers/body from all pages
- Runs `ReadingOrderModel` to establish document structure
- Generates page/picture/table images based on configuration
- Computes confidence scores (layout, parse, table, OCR)

**Batch Configuration:**
```python
ocr_batch_size: int = 4
layout_batch_size: int = 4
table_batch_size: int = 4
queue_max_size: int = 100
batch_polling_interval_seconds: float = 0.5
```

**Performance Characteristics:**
- Parallel stage execution with backpressure control
- Configurable queue sizes to balance memory vs throughput
- Thread-safe resource management
- Graceful timeout handling

---

### 1.3 VLM Pipeline

**Location:** `docling/pipeline/vlm_pipeline.py`

#### VlmPipeline

Process documents using Vision-Language Models (VLMs).

**Supported VLM Types:**
- **API-based:** External services (OpenAI-compatible APIs)
- **Inline models:**
  - HuggingFace Transformers
  - MLX (Apple Silicon optimized)
  - VLLM (high-throughput inference)

**Response Formats:**
1. **DOCTAGS**: Structured document tags with bounding boxes
2. **MARKDOWN**: Generated markdown representation
3. **HTML**: Generated HTML representation

**Key Features:**
- `force_backend_text`: Option to use PDF backend text instead of VLM-generated text
- Page image generation
- Picture image extraction
- Multi-framework support

**Processing Flow:**
1. Initialize page with backend
2. Run VLM model on page image
3. Convert VLM response to `DoclingDocument`:
   - **DOCTAGS**: Parse structured tags and create document
   - **MARKDOWN/HTML**: Convert to intermediate format, then to `DoclingDocument`
4. Optionally extract text from backend using predicted bounding boxes
5. Generate picture images if configured

**Use Cases:**
- Processing scanned documents without OCR
- Layout-free document understanding
- Quick document digitization
- Multi-language document processing

---

### 1.4 Extraction VLM Pipeline

**Location:** `docling/pipeline/extraction_vlm_pipeline.py`

#### ExtractionVlmPipeline

**Extends:** `BaseExtractionPipeline`

**Purpose:** Extract structured data from documents using VLMs

**Model:** Uses `NuExtractTransformersModel`

**Key Features:**
- Template-based extraction (string, dict, or Pydantic `BaseModel`)
- Per-page extraction results
- JSON parsing of extracted data
- Respects page range limits

**Extraction Flow:**
1. Get images from input document via backend
2. Serialize template (if provided)
3. Process each image with VLM using template as prompt
4. Parse JSON responses (if valid)
5. Create `ExtractedPageData` for each page
6. Determine overall status

**Template Types:**
```python
ExtractionTemplateType = str | Dict | BaseModel | Type[BaseModel]
```

**Use Cases:**
- Form data extraction
- Invoice/receipt parsing
- Contract information extraction
- Structured metadata extraction

---

### 1.5 Simple Pipeline

**Location:** `docling/pipeline/simple_pipeline.py`

#### SimplePipeline

**Purpose:** Handle formats with declarative backends that produce `DoclingDocument` directly

**Supported Formats:** DOCX, PPTX, HTML, Markdown, CSV, XLSX, XML variants

**Processing:**
- No page-level pipeline needed
- Backend directly converts to `DoclingDocument`
- Used for structured formats that don't require AI models

**Characteristics:**
- Fastest processing (no AI models)
- Direct format-to-document conversion
- Minimal overhead

---

### 1.6 ASR Pipeline

**Location:** `docling/pipeline/asr_pipeline.py`

#### AsrPipeline

**Purpose:** Audio/video transcription using Whisper models

**Supported Models:**
- Native Whisper (OpenAI)
- MLX Whisper (Apple Silicon optimized)

**Features:**
- Word-level timestamps
- Segment-level timestamps
- Speaker identification support
- Temporary file handling for streams
- Multiple audio/video format support

**Output Structure:**
- `ConversationItem` with text, timestamps, speaker info
- Word-level granularity (optional)
- Formatted as `DoclingDocument` with TEXT items

**Processing Flow:**
1. Load audio file (or create temp file from stream)
2. Run Whisper transcription
3. Extract segments with timestamps
4. Optional word-level timestamps
5. Create `DoclingDocument` with TEXT items
6. Format: `"[time: start-end] [speaker:name] text"`

---

## 2. Data Models

### 2.1 Core Document Models

**Location:** `docling/datamodel/document.py`

#### InputDocument

Represents a document as input to conversion.

**Key Fields:**
```python
file: Path                      # Path representation
document_hash: str             # Stable hash of content
valid: bool                    # Validation status
format: InputFormat            # Format enum (PDF, DOCX, etc.)
page_count: int                # Number of pages (for paginated docs)
filesize: int                  # Size in bytes
limits: DocumentLimits         # Page range, max size, max pages
backend_options: Any           # Custom backend configuration
_backend: AbstractDocumentBackend  # Internal backend instance
```

**Validation:**
- File size limits
- Page count limits
- Backend validity
- Page range validation

**Methods:**
```python
def valid_from(self, page: int) -> int:
    """First valid page number"""

def valid_to(self, page: int) -> int:
    """Last valid page number"""
```

#### ConversionResult

Complete result of document conversion.

**Key Fields:**
```python
input: InputDocument           # Original input
status: ConversionStatus       # SUCCESS, PARTIAL_SUCCESS, FAILURE
errors: List[ErrorItem]        # Conversion errors
pages: List[Page]              # Page-level results
assembled: AssembledUnit       # Elements, headers, body
timings: Dict                  # Profiling information
confidence: ConfidenceReport   # Quality scores
document: DoclingDocument      # Final output
```

**Status Enum:**
- `SUCCESS`: Full conversion completed
- `PARTIAL_SUCCESS`: Some pages/elements failed
- `FAILURE`: Conversion failed entirely

---

### 2.2 Base Models

**Location:** `docling/datamodel/base_models.py`

#### Page

Represents a single document page.

**Key Fields:**
```python
page_no: int                   # Page number (0-indexed)
size: Size                     # (width, height)
parsed_page: SegmentedPdfPage  # From backend
predictions: PagePredictions   # Layout, tables, figures, VLM
assembled: AssembledUnit       # Page elements
_backend: PageBackend          # Page backend instance
_image_cache: Dict             # Cached images at different scales
```

**Methods:**
```python
def get_image(self, scale: float = 1.0,
              cropbox: BoundingBox = None) -> PIL.Image:
    """Get page image at specified scale/cropbox"""

@property
def cells(self) -> List[TextCell]:
    """Text cells from parsed_page"""
```

#### Cluster

Represents a predicted layout cluster.

**Fields:**
```python
id: int                        # Cluster identifier
label: DocItemLabel            # TEXT, TABLE, PICTURE, etc.
bbox: BoundingBox              # Bounding box
confidence: float              # Prediction confidence
cells: List[TextCell]          # Text cells in cluster
children: List[Cluster]        # Child clusters (hierarchical)
```

**Label Types:**
- `TEXT`: Regular text paragraphs
- `TABLE`: Tables
- `PICTURE`: Images, figures
- `FORMULA`: Mathematical formulas
- `CODE`: Code blocks
- `LIST_ITEM`: List items
- `CAPTION`: Captions
- `FOOTNOTE`: Footnotes
- `PAGE_HEADER`, `PAGE_FOOTER`: Headers/footers
- `FORM`, `KEY_VALUE_REGION`: Form elements

#### PagePredictions

Aggregates all predictions for a page.

**Fields:**
```python
layout: LayoutPrediction                      # Clusters
tablestructure: TableStructurePrediction      # Table cells
figures_classification: FigureClassificationPrediction
equations_prediction: EquationPrediction
vlm_response: VlmPrediction
```

#### ConfidenceReport

Quality scores for document conversion.

**Scores:**
```python
parse_score: float      # Backend parsing quality [0-1]
layout_score: float     # Layout detection quality [0-1]
table_score: float      # Table structure quality [0-1]
ocr_score: float        # OCR quality [0-1]
mean_score: float       # Average across metrics
low_score: float        # 5th percentile (worst quality indicator)
```

**Grades:**
- `POOR`: < 0.5
- `FAIR`: 0.5 - 0.7
- `GOOD`: 0.7 - 0.9
- `EXCELLENT`: > 0.9
- `UNSPECIFIED`: No data

**Usage:** Indicates overall conversion quality for downstream decisions

---

### 2.3 Pipeline Options

**Location:** `docling/datamodel/pipeline_options.py`

#### PipelineOptions (Base)

Base configuration for all pipelines.

```python
document_timeout: Optional[float]              # Timeout in seconds
accelerator_options: AcceleratorOptions        # Device and threading
enable_remote_services: bool = False           # Allow API-based services
allow_external_plugins: bool = False           # Load external plugins
artifacts_path: Optional[Path] = None          # Model storage location
```

#### PdfPipelineOptions

Comprehensive PDF processing configuration.

**Layout Options:**
```python
do_table_structure: bool = True        # Enable table extraction
do_ocr: bool = True                   # Enable OCR
do_code_enrichment: bool = True       # Code block detection
do_formula_enrichment: bool = True    # Formula extraction (LaTeX)
layout_options: LayoutOptions         # Model spec, cell assignment
```

**OCR Options:**
```python
ocr_options: OcrOptions
  - engine: OcrEngine                 # auto, tesseract, easyocr, rapidocr, ocrmac
  - lang: List[str]                   # Language codes
  - force_full_page_ocr: bool = False # OCR entire page
  - bitmap_area_threshold: float = 0.05  # Min bitmap area for OCR
```

**Table Options:**
```python
table_structure_options: TableStructureOptions
  - do_cell_matching: bool = True     # Match predictions to PDF cells
  - mode: TableFormerMode = FAST      # FAST or ACCURATE
```

**Image Generation:**
```python
images_scale: float = 1.0             # Image resolution scaling
generate_page_images: bool = False     # Full page images
generate_picture_images: bool = True   # Cropped picture images
generate_table_images: bool = False    # Cropped table images (deprecated)
generate_parsed_pages: bool = False    # Keep parsed page data
```

**Threading Options:**
```python
ocr_batch_size: int = 4
layout_batch_size: int = 4
table_batch_size: int = 4
queue_max_size: int = 100              # Backpressure control
batch_polling_interval_seconds: float = 0.5
```

#### VlmPipelineOptions

Vision-Language Model configuration.

```python
vlm_options: InlineVlmOptions | ApiVlmOptions
  # Inline options:
  - repo_id: str                      # HuggingFace model ID
  - inference: VlmInferenceFramework  # TRANSFORMERS, MLX, VLLM
  - generation_config: Dict           # max_tokens, temperature, etc.
  - prompt_style: VlmPromptStyle      # RAW, NONE, CHAT
  - response_format: VlmResponseFormat # DOCTAGS, MARKDOWN, HTML

  # API options:
  - api_url: str                      # Endpoint URL
  - headers: Dict                     # Auth headers
  - params: Dict                      # Request params

force_backend_text: bool = False      # Use backend text instead of VLM text
```

#### AsrPipelineOptions

Audio transcription configuration.

```python
asr_options: InlineAsrOptions
  - model: str                        # Whisper model variant
  - timestamps: bool = True           # Segment timestamps
  - word_timestamps: bool = False     # Word-level timestamps
  - language: Optional[str]           # Language code
  - task: str = "transcribe"          # "transcribe" or "translate"
  - temperature: float = 0.0          # Sampling temperature
  - compression_ratio_threshold: float
  - logprob_threshold: float
  - no_speech_threshold: float
```

---

### 2.4 Extraction Models

**Location:** `docling/datamodel/extraction.py`

#### ExtractedPageData

Per-page extraction result.

```python
page_no: int                    # 1-indexed page number
extracted_data: Dict            # Structured data dict
raw_text: str                   # Raw text from VLM
errors: List[ErrorItem]         # Per-page errors
```

#### ExtractionResult

Complete extraction result.

```python
input: InputDocument
status: ConversionStatus
errors: List[ErrorItem]
pages: List[ExtractedPageData]
```

#### ExtractionTemplateType

Flexible template specification.

```python
ExtractionTemplateType = str | Dict | BaseModel | Type[BaseModel]
```

**Examples:**
```python
# String template
template = "Extract invoice number, date, and total amount"

# Dict template
template = {"invoice_number": "str", "date": "str", "total": "float"}

# Pydantic model
class Invoice(BaseModel):
    invoice_number: str
    date: datetime
    total: float
template = Invoice
```

---

## 3. Backend & Converters

### 3.1 Abstract Backend

**Location:** `docling/backend/abstract_backend.py`

#### AbstractDocumentBackend (ABC)

Interface for all document backends.

**Key Methods:**
```python
def is_valid(self) -> bool:
    """Check if backend initialized correctly"""

def supports_pagination(self) -> bool:
    """Whether backend handles paginated docs"""

@classmethod
def supported_formats(cls) -> Set[InputFormat]:
    """Set of supported InputFormat"""

def unload(self):
    """Cleanup resources"""
```

#### PaginatedDocumentBackend

**Extends:** `AbstractDocumentBackend`

**Additional Method:**
```python
def page_count(self) -> int:
    """Total number of pages"""
```

**Used by:** PDF, Image backends

#### DeclarativeDocumentBackend

**Extends:** `AbstractDocumentBackend`

**Purpose:** Backends that directly output `DoclingDocument`

**Method:**
```python
def convert(self) -> DoclingDocument:
    """Direct conversion to DoclingDocument"""
```

**Used by:** DOCX, HTML, Markdown, CSV, XLSX, XML backends

---

### 3.2 Backend Implementations

**Available Backends:**

1. **PDF Backends:**
   - `DoclingParseV4DocumentBackend`: Modern PDF parser (default)
   - `DoclingParseV2DocumentBackend`: Legacy parser
   - `PypdfiumBackend`: Alternative PDF backend

2. **Office Formats:**
   - `MsWordDocumentBackend`: DOCX processing
   - `MsExcelDocumentBackend`: XLSX processing
   - `MsPowerpointDocumentBackend`: PPTX processing

3. **Web Formats:**
   - `HTMLDocumentBackend`: HTML parsing
   - `MarkdownDocumentBackend`: Markdown parsing

4. **Structured Formats:**
   - `CsvDocumentBackend`: CSV files
   - `DoclingJSONBackend`: Docling JSON format

5. **Scientific Formats:**
   - `PatentUsptoDocumentBackend`: USPTO patent XML
   - `JatsDocumentBackend`: JATS scientific XML
   - `MetsGbsDocumentBackend`: Google Books METS

6. **Media Formats:**
   - `NoOpBackend`: Used for audio (no document structure)
   - `WebVTTDocumentBackend`: Subtitle files

7. **Other:**
   - `AsciiDocBackend`: AsciiDoc format

**Backend Selection:**
- Automatic based on `InputFormat`
- Configurable via `backend_options`
- Extensible via plugin system

---

### 3.3 Document Converter

**Location:** `docling/document_converter.py`

#### DocumentConverter

Main entry point for document conversion.

**Key Responsibilities:**
- Format detection and routing
- Pipeline initialization and caching
- Batch processing
- Error handling

**Configuration:**
```python
allowed_formats: List[InputFormat]          # Formats to support
format_options: Dict[InputFormat, FormatOption]  # Custom per format
initialized_pipelines: Dict                 # Cached pipelines
```

**Key Methods:**

```python
def convert(self, source: Path | str | DocumentStream) -> ConversionResult:
    """Convert single document"""

def convert_all(self, sources: Iterable,
                doc_batch_size: int = 1,
                doc_batch_concurrency: int = 1) -> Iterator[ConversionResult]:
    """Batch conversion with iterator"""

def convert_string(self, content: str,
                   format: InputFormat) -> ConversionResult:
    """Convert string content (MD/HTML)"""

def initialize_pipeline(self, format: InputFormat):
    """Pre-initialize pipeline for format"""
```

**Processing Flow:**
1. Accept `Path`, str (URL), or `DocumentStream`
2. Guess document format (MIME type, extension, content analysis)
3. Get or initialize appropriate pipeline
4. Execute pipeline on `InputDocument`
5. Return `ConversionResult`

**FormatOption Classes:**

Map `InputFormat` to (Backend, Pipeline, Options):
```python
class FormatOption:
    backend: Type[AbstractDocumentBackend]
    pipeline: Type[BasePipeline]
    pipeline_options: PipelineOptions
```

**Pipeline Caching:**
- Pipelines cached by `(class, options_hash)`
- Reused across conversions with same options
- Thread-safe cache locking
- Reduces model initialization overhead

**Batch Processing:**
```python
converter.convert_all(
    sources=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    doc_batch_size=10,           # Process 10 at a time
    doc_batch_concurrency=4      # 4 parallel threads
)
```

**Format Detection:**
- MIME type sniffing
- File extension matching
- Content-based detection
- Manual override via `InputFormat`

---

## 4. Models & Processing

### 4.1 Base Model Classes

**Location:** `docling/models/base_model.py`

#### BasePageModel (ABC)

Interface for page-level processing models.

```python
def __call__(self, conv_res: ConversionResult,
             page_batch: Iterable[Page]) -> Iterable[Page]:
    """Process batch of pages"""
```

**Used by:** Layout, OCR, Table, Assemble models

#### BaseVlmModel (ABC)

Interface for Vision-Language Models.

```python
def process_images(self, image_batch: List[PIL.Image],
                   prompt: str | List[str]) -> Iterable[VlmPrediction]:
    """Process images with VLM"""
```

**Supports:** Single prompt or list of prompts

#### BaseVlmPageModel

**Combines:** `BasePageModel` + `BaseVlmModel`

**Additional:**
```python
def formulate_prompt(self, prompt_style: VlmPromptStyle) -> str:
    """Generate model-specific prompt"""
```

**Prompt Styles:**
- `RAW`: Plain text prompt
- `NONE`: No prompt
- `CHAT`: Conversational format
- Model-specific styles

#### GenericEnrichmentModel (ABC)

Base for document enrichment (post-conversion).

**Methods:**
```python
def is_processable(self, item: NodeItem) -> bool:
    """Check if element should be processed"""

def prepare_element(self, item: NodeItem, page: Page) -> Any:
    """Prepare element for processing"""

def __call__(self, doc: DoclingDocument) -> DoclingDocument:
    """Process batch of elements"""
```

**Configuration:**
- `elements_batch_size`: Configurable batch size

#### BaseEnrichmentModel

Generic enrichment on `NodeItem`.

#### BaseItemAndImageEnrichmentModel

Image-based enrichment (cropping from pages).

**Used for:** Picture description, classification

**Features:**
- Bounding box expansion
- Image caching
- Batch processing

---

### 4.2 Layout Model

**Location:** `docling/models/layout_model.py`

#### LayoutModel

Detect document structure using deep learning.

**Backend:** `docling_ibm_models.LayoutPredictor`

**Model:** Configurable (Heron, Egret variants)

**Detected Elements:**

- **TEXT_ELEM_LABELS:** Text, footnotes, captions, checkboxes, headers, footers, code, list items, formulas
- **TABLE_LABELS:** Tables, document indices
- **FIGURE_LABEL:** Pictures
- **FORMULA_LABEL:** Mathematical formulas
- **CONTAINER_LABELS:** Forms, key-value regions

**Processing:**
1. Extract page images (scale 1.0)
2. Batch predict layouts for all pages
3. Convert predictions to `Cluster` objects
4. Apply `LayoutPostprocessor`:
   - Cell-to-cluster assignment
   - Orphan cluster creation
   - Empty cluster filtering
   - Child cluster support
5. Compute layout and OCR confidence scores
6. Store in `page.predictions.layout`

**Postprocessing Options:**
```python
create_orphan_clusters: bool = False   # Create clusters for unassigned cells
keep_empty_clusters: bool = False      # Retain clusters without text
skip_cell_assignment: bool = False     # Skip for VLM-only processing
```

**Debugging:**
- Visualize raw and postprocessed layouts
- Side-by-side cluster visualization

**Model Variants:**
- **Heron:** Faster, default
- **Egret:** More accurate, slower

---

### 4.3 Table Structure Model

**Location:** `docling/models/table_structure_model.py`

#### TableStructureModel

Extract table structure and cells.

**Backend:** `docling_ibm_models.TableFormer`

**Modes:**
- `FAST`: Faster inference
- `ACCURATE`: Higher quality, slower

**Processing:**
1. Find table clusters from layout predictions
2. Scale page image to 2x (144 DPI)
3. Extract text cells within table bounding box
4. Run TableFormer prediction
5. Match predicted cells to PDF cells (optional)
6. Extract text from backend if no cell matching
7. Create `Table` objects with:
   - OTSL sequence (table structure)
   - `TableCell` list with positions
   - Row/column counts

**Options:**
```python
do_cell_matching: bool = True
```
- **True:** Use PDF cell structure (can break if cells merged across columns)
- **False:** Use TableFormer cell definitions

**Output:**
- `TableStructurePrediction` with `table_map`
- Each `Table` has structured cells, OTSL sequence, dimensions

**OTSL (Object Table Structure Language):**
- Sequence representation of table structure
- Used for downstream processing

---

### 4.4 Page Assembly Model

**Location:** `docling/models/page_assemble_model.py`

#### PageAssembleModel

Assemble page elements from clusters and predictions.

**Input:**
- Layout clusters
- Table predictions
- Figure predictions

**Output:**
- `AssembledUnit` (elements, headers, body)

**Text Processing:**
- Sanitize text: Handle hyphenation, normalize characters
- Join text cells from clusters
- Remove control characters

**Element Creation:**
1. **TextElement:** From TEXT_ELEM_LABELS clusters
2. **Table:** From table predictions or fallback
3. **FigureElement:** From figure predictions or fallback
4. **ContainerElement:** Forms, key-value regions

**Organization:**
```python
class AssembledUnit:
    elements: List[DocElement]    # All page elements
    headers: List[DocElement]     # Page headers/footers
    body: List[DocElement]        # Main content elements
```

**Fallback Handling:**
- If no table prediction, create table from cluster
- If no figure prediction, create figure from cluster

---

### 4.5 Reading Order Model

**Location:** `docling/models/readingorder_model.py`

#### ReadingOrderModel

Establish reading order and create final `DoclingDocument`.

**Backend:** `docling_ibm_models.ReadingOrderPredictor`

**Processing Steps:**
1. Convert assembled elements to ReadingOrder format
2. Predict reading order (sort elements)
3. Predict caption relationships
4. Predict footnote relationships
5. Predict element merges
6. Build `DoclingDocument` with proper hierarchy

**Document Structure:**
- Pages with metadata (size, images)
- Text items (paragraphs, headers, lists)
- Tables with structure and cells
- Pictures with captions/footnotes
- Code blocks
- Formulas (text + LaTeX original)
- Groups (lists, forms, key-value areas)

**Special Handling:**
- List item processing (markers, numbering)
- Rich table cells (with nested content)
- Caption/footnote attachment
- Element merging across columns/pages
- Hierarchical child elements

**Reading Order Algorithm:**
- Spatial analysis (top-to-bottom, left-to-right)
- Column detection
- Multi-column flow handling
- Page break handling

---

### 4.6 OCR Models

**Factory-based instantiation** via `get_ocr_factory()`

**Available Engines:**

1. **Auto:** Selects best available engine
2. **RapidOCR:** Fast, supports English/Chinese
3. **EasyOCR:** Multi-language support
4. **Tesseract:** Popular open-source OCR
5. **TesseractCLI:** Command-line wrapper
6. **OCRMac:** Native macOS OCR

**Configuration:**
```python
class OcrOptions:
    engine: OcrEngine              # Engine selection
    lang: List[str]                # Language codes
    force_full_page_ocr: bool      # Force OCR entire page
    bitmap_area_threshold: float   # Min bitmap area for OCR
```

**OCR Triggers:**
- Scanned PDF pages (no text layer)
- Bitmap areas exceeding threshold
- Force full-page OCR enabled
- Missing text in layout clusters

**Multi-language Support:**
- Language codes: "eng", "fra", "deu", "chi_sim", etc.
- Multiple languages per document
- Automatic language detection (some engines)

---

### 4.7 VLM Models

**Location:** `docling/models/vlm_models_inline/`

#### 1. HuggingFaceTransformersVlmModel

Uses Transformers library for VLM inference.

**Supported Architectures:**
- Qwen2-VL
- Florence-2
- Granite-Docling
- Custom models

**Features:**
- Prompt templating
- Generation config (temperature, max_tokens, etc.)
- Response format handling

#### 2. VllmVlmModel

Uses VLLM for high-throughput inference.

**Features:**
- Optimized batching
- KV cache optimization
- Parallel sampling
- High-throughput serving

#### 3. HuggingFaceMlxModel

Apple Silicon optimized (MLX framework).

**Features:**
- Memory efficient
- Fast inference on M-series chips
- Native Metal acceleration

#### 4. NuExtractTransformersModel

Specialized for structured extraction.

**Used in:** `ExtractionVlmPipeline`

**Features:**
- Template-based extraction
- JSON output parsing
- Pydantic model support

#### ApiVlmModel

OpenAI-compatible API client.

**Features:**
- Remote VLM services
- Configurable endpoints
- Custom headers and params
- Supports multiple API providers

**Configuration:**
```python
class ApiVlmOptions:
    api_url: str                   # Endpoint URL
    headers: Dict                  # Auth headers
    params: Dict                   # Request params
    model: str                     # Model name
```

---

### 4.8 Other Models

#### CodeFormulaModel

Extract code blocks and formulas from text.

**Features:**
- Code block detection (programming languages)
- LaTeX formula extraction
- Inline and display formulas
- Code language identification

#### DocumentPictureClassifier

Classify picture types.

**Categories:**
- Photograph
- Diagram
- Chart
- Flowchart
- Graph
- Illustration
- Logo
- Icon

#### PictureDescriptionVlmModel

Generate natural language descriptions of pictures.

**Features:**
- VLM-based captioning
- Context-aware descriptions
- Configurable detail level

#### PagePreprocessingModel

Initial page processing.

**Tasks:**
- Image quality checks
- Resolution normalization
- Color space conversion
- Page rotation detection

---

### 4.9 Model Factories

**Location:** `docling/models/factories/base_factory.py`

#### BaseFactory

Plugin-based model registration and instantiation.

**Features:**
- Dynamic plugin loading
- Option-based instantiation
- External plugin support (configurable)
- Registration validation

**Available Factories:**
1. `get_ocr_factory()`: OCR engine selection
2. `get_picture_description_factory()`: Picture description models

**Plugin System:**
- Uses `pluggy` for plugin management
- Entry points: "docling"
- External plugins can be loaded if allowed
- Metadata tracking (plugin name, module)

**Factory Pattern:**
```python
factory = get_ocr_factory()
ocr_model = factory.get_model(ocr_options)
```

---

## 5. Chunking & Post-processing

### 5.1 Chunking

**Location:** `docling/chunking/`

**Available Chunkers** (from docling-core):

#### 1. BaseChunker

Abstract base class for all chunkers.

#### 2. HierarchicalChunker

Respects document hierarchy when chunking.

**Features:**
- `DocChunk` with metadata
- Preserves structure (sections, paragraphs)
- Parent-child relationships

**Configuration:**
```python
class HierarchicalChunkerOptions:
    max_chunk_size: int            # Max tokens per chunk
    overlap: int                   # Token overlap between chunks
    respect_headers: bool = True   # Keep headers with content
```

#### 3. HybridChunker

Combines multiple chunking strategies.

**Strategies:**
- Sentence-based
- Token-based
- Paragraph-based
- Semantic-based

**Configuration:**
```python
class HybridChunkerOptions:
    max_chunk_size: int
    overlap: int
    strategy: ChunkingStrategy
```

**Purpose:** Split documents into chunks for:
- RAG (Retrieval Augmented Generation)
- Embedding generation
- Search indexing
- Vector database storage

**Output Format:**
```python
class DocChunk:
    text: str                      # Chunk text
    metadata: Dict                 # Page number, position, etc.
    doc_items: List[NodeItem]      # Source document items
```

---

### 5.2 Post-processing Components

#### Layout Postprocessor

**Location:** `docling/models/layout_model.py`

**Responsibilities:**
- Cell-to-cluster assignment
- Orphan cluster handling
- Cluster filtering
- Text cell normalization

**Processing:**
1. Assign PDF text cells to layout clusters
2. Create orphan clusters for unassigned cells
3. Filter empty clusters (optional)
4. Normalize cluster text

#### List Item Processor

**Location:** `docling/models/readingorder_model.py`

**Responsibilities:**
- Marker detection (bullets, numbers)
- List type inference (ordered/unordered)
- Item normalization

**Detected Markers:**
- Bullets: •, ◦, ▪, ▫, –, *, etc.
- Numbers: 1., 2., 3., etc.
- Letters: a., b., c., etc.
- Roman numerals: i., ii., iii., etc.

#### Text Sanitization

**Responsibilities:**
- Hyphenation handling (join split words)
- Character normalization (Unicode)
- Whitespace cleanup
- Control character removal

**Hyphenation Rules:**
- End-of-line hyphens removed
- Words rejoined across lines
- Preserves intentional hyphens

---

## 6. Document Processing Flow

### 6.1 PDF Document Flow (Standard Pipeline)

**Complete end-to-end flow:**

#### 1. Input & Initialization
- `DocumentConverter` receives `Path`/`Stream`
- Format detection (MIME type, extension)
- `InputDocument` creation with `PdfDocumentBackend`
- Validation (file size, page count, limits)

#### 2. Pipeline Selection
- `StandardPdfPipeline` selected for PDF format
- Pipeline cached or reused based on options hash
- Models initialized (Layout, OCR, Table, etc.)

#### 3. Build Phase (Multi-threaded)

**Preprocess Stage:**
- Load page backends lazily
- Initialize page size
- Run `PagePreprocessingModel`

**OCR Stage:**
- Extract text from images/bitmaps
- Batch size: 4 pages
- Parallel processing

**Layout Stage:**
- Detect document structure
- Batch size: 4 pages
- Create clusters (text, tables, figures)

**Table Stage:**
- Extract table structure
- Batch size: 4 pages
- Run TableFormer

**Assemble Stage:**
- Create page elements
- Release page resources
- Batch size: 1 page

**Threading:**
- 5 parallel stages with bounded queues
- Backpressure control (queue size: 100)
- Polling interval: 0.5s

#### 4. Assembly Phase
- Aggregate elements from all pages
- Run `ReadingOrderModel`:
  - Sort elements in reading order
  - Attach captions/footnotes
  - Merge split elements
- Create `DoclingDocument` with hierarchy
- Generate page/picture/table images if configured
- Compute confidence scores

#### 5. Enrichment Phase
- **CodeFormulaModel:** Extract code and formulas (optional)
- **DocumentPictureClassifier:** Classify pictures (optional)
- **PictureDescriptionVlmModel:** Describe pictures (optional)
- Process in batches (default batch size from settings)

#### 6. Finalization
- Determine final status (SUCCESS, PARTIAL_SUCCESS, FAILURE)
- Cleanup backends and images
- Return `ConversionResult` with `DoclingDocument`

**Timing:** ~2-10 seconds per page depending on configuration

---

### 6.2 VLM Document Flow

**Vision-Language Model processing:**

#### 1. Input & Initialization
- Same as PDF flow
- `VlmPipeline` selected based on `vlm_options`

#### 2. Build Phase
- Initialize pages with backends
- Extract page images at configured scale
- Run VLM model (API or inline) with prompt
- Collect `VlmPrediction` per page

#### 3. Assembly Phase
- Convert VLM responses to `DoclingDocument`:
  - **DOCTAGS:** Parse structured tags, create hierarchy
  - **MARKDOWN:** Convert MD to `DoclingDocument` via backend
  - **HTML:** Convert HTML to `DoclingDocument` via backend
- Optionally extract text from PDF backend using predicted boxes
- Generate picture images if configured

#### 4. Enrichment Phase
- Same enrichment models as PDF pipeline

#### 5. Finalization
- Same as PDF flow

**Timing:** ~5-30 seconds per page depending on VLM and hardware

---

### 6.3 Extraction Flow

**Structured data extraction:**

#### 1. Input & Initialization
- `InputDocument` with `PdfDocumentBackend`
- `ExtractionVlmPipeline` with template

#### 2. Extraction Phase
- Get images from backend (respects page range)
- Serialize template to prompt
- Process each image with `NuExtractTransformersModel`
- Parse JSON responses

#### 3. Result Collection
- Create `ExtractedPageData` per page
- Populate `extracted_data` (dict) or `raw_text` (string)
- Track errors per page

#### 4. Finalization
- Return `ExtractionResult` with pages

**Timing:** ~3-15 seconds per page depending on template complexity

---

### 6.4 Declarative Document Flow (DOCX, HTML, etc.)

**Simple format conversion:**

#### 1. Input & Initialization
- Format-specific `DeclarativeDocumentBackend`
- `SimplePipeline` selected

#### 2. Build Phase
- `Backend.convert()` called directly
- Returns complete `DoclingDocument`
- No page-level pipeline needed

#### 3. Enrichment Phase
- Same enrichment models available

#### 4. Finalization
- Return `ConversionResult` with document

**Timing:** < 1 second per document (fast)

---

### 6.5 Audio/Video Flow

**Transcription processing:**

#### 1. Input & Initialization
- `NoOpBackend` (no document structure)
- `AsrPipeline` with Whisper model

#### 2. Transcription Phase
- Load audio file (or create temp file from stream)
- Run Whisper transcription
- Extract segments with timestamps
- Optional word-level timestamps

#### 3. Document Creation
- Create `DoclingDocument`
- Add TEXT items for each segment
- Include timestamps and speaker info
- Format: `"[time: start-end] [speaker:name] text"`

#### 4. Finalization
- Cleanup temp files
- Return `ConversionResult`

**Timing:** ~0.1-0.5x real-time depending on model and hardware

---

## 7. Configuration & Customization

### 7.1 Pipeline Configuration

#### Device Selection

```python
from docling.datamodel.accelerator_options import AcceleratorDevice

accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA,  # CPU, CUDA, MPS, ROCM
    num_threads=4                   # For CPU inference
)
```

#### Timeout Configuration

```python
pipeline_options = PdfPipelineOptions(
    document_timeout=300.0  # 5 minutes per document
)
```

#### Artifact Paths

```python
pipeline_options = PdfPipelineOptions(
    artifacts_path=Path("/custom/models/")
)
```

---

### 7.2 Backend Options

#### PDF Backend Options

```python
from docling.datamodel.backend_options import PdfBackendOptions

backend_options = PdfBackendOptions(
    backend="pypdfium2"  # or "dlparse_v2", "dlparse_v4"
)
```

#### Markdown Backend Options

```python
from docling.datamodel.backend_options import MarkdownBackendOptions

backend_options = MarkdownBackendOptions(
    flavor="github"  # or "commonmark", "markdown_it"
)
```

---

### 7.3 Model Options

#### Layout Options

```python
from docling.datamodel.layout_model_specs import LayoutModelOptions

layout_options = LayoutModelOptions(
    model_spec="heron",           # or "egret"
    create_orphan_clusters=False,
    keep_empty_clusters=False,
    skip_cell_assignment=False
)
```

#### Table Structure Options

```python
from docling.datamodel.table_structure_options import TableStructureOptions

table_options = TableStructureOptions(
    do_cell_matching=True,
    mode=TableFormerMode.ACCURATE  # or FAST
)
```

#### OCR Options

```python
from docling.datamodel.ocr_options import OcrOptions, OcrEngine

ocr_options = OcrOptions(
    engine=OcrEngine.EASYOCR,     # or TESSERACT, RAPIDOCR, etc.
    lang=["eng", "fra"],          # Multiple languages
    force_full_page_ocr=False,
    bitmap_area_threshold=0.05
)
```

#### VLM Options

```python
from docling.datamodel.vlm_model_specs import InlineVlmOptions

vlm_options = InlineVlmOptions(
    repo_id="ibm-granite/granite-docling-258M",
    inference=VlmInferenceFramework.TRANSFORMERS,
    generation_config={
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    },
    prompt_style=VlmPromptStyle.CHAT,
    response_format=VlmResponseFormat.DOCTAGS
)
```

---

### 7.4 Performance Tuning

#### Batch Sizes

```python
pdf_options = PdfPipelineOptions(
    ocr_batch_size=8,           # Increase for more throughput
    layout_batch_size=8,
    table_batch_size=8,
    queue_max_size=200          # Increase for more buffering
)
```

#### Concurrency

```python
converter = DocumentConverter()
results = converter.convert_all(
    sources=documents,
    doc_batch_size=20,           # Documents per batch
    doc_batch_concurrency=8      # Parallel threads
)
```

#### Memory Management

```python
pdf_options = PdfPipelineOptions(
    generate_parsed_pages=False,   # Don't keep parsed pages
    generate_page_images=False,    # Don't generate page images
    generate_picture_images=True   # Only generate picture images
)
```

#### Timeouts

```python
pdf_options = PdfPipelineOptions(
    document_timeout=600.0,           # 10 minutes per document
    batch_polling_interval_seconds=0.1  # Faster polling
)
```

---

## 8. Key Design Patterns

### 8.1 Factory Pattern

**Usage:** Model instantiation

```python
# OCR factory
ocr_factory = get_ocr_factory()
ocr_model = ocr_factory.get_model(ocr_options)

# Picture description factory
pic_desc_factory = get_picture_description_factory()
pic_desc_model = pic_desc_factory.get_model(options)
```

**Benefits:**
- Pluggable model implementations
- External plugin support
- Clean instantiation API

---

### 8.2 Pipeline Pattern

**Usage:** Multi-stage document processing

```python
# Pipeline stages
preprocess -> ocr -> layout -> table -> assemble
```

**Benefits:**
- Clear separation of concerns
- Composable stages
- Parallel execution

---

### 8.3 Strategy Pattern

**Usage:** Backend selection, pipeline selection

```python
# Backend strategy
format -> backend_class

# Pipeline strategy
format + options -> pipeline_class
```

**Benefits:**
- Flexible format support
- Easy to add new formats
- Runtime selection

---

### 8.4 Iterator Pattern

**Usage:** Lazy processing, batch processing

```python
# Lazy page processing
for page in pages:
    process(page)

# Batch conversion
for result in converter.convert_all(documents):
    handle(result)
```

**Benefits:**
- Memory efficient
- Streaming support
- Progress tracking

---

### 8.5 Builder Pattern

**Usage:** ConversionResult construction

```python
result = ConversionResult(
    input=input_doc,
    status=ConversionStatus.SUCCESS,
    document=docling_document,
    confidence=confidence_report,
    timings=timings
)
```

**Benefits:**
- Incremental construction
- Clear state management
- Immutable results

---

### 8.6 Template Method Pattern

**Usage:** BasePipeline workflow

```python
class BasePipeline(ABC):
    def execute(self, in_doc):
        self._build_document(in_doc)
        self._assemble_document(in_doc)
        self._enrich_document(in_doc)
        return result

    @abstractmethod
    def _build_document(self, in_doc):
        pass
```

**Benefits:**
- Consistent workflow
- Extensible behavior
- Code reuse

---

### 8.7 Observer Pattern (Implicit)

**Usage:** Profiling, confidence scoring

```python
# Timings collected throughout pipeline
conv_res.timings["layout"] = layout_time
conv_res.timings["ocr"] = ocr_time

# Confidence scores aggregated
conf_report.layout_score = mean(page_layout_scores)
```

**Benefits:**
- Automatic metrics collection
- No coupling between stages
- Easy monitoring

---

## 9. Dependencies & Relationships

### 9.1 Core Dependencies

**Document Flow:**
```
DocumentConverter -> Pipeline -> Backend -> Models -> DoclingDocument
```

**Pipeline Hierarchy:**
```
BasePipeline
├── ConvertPipeline
│   ├── PaginatedPipeline
│   │   ├── StandardPdfPipeline
│   │   └── VlmPipeline
│   └── SimplePipeline
└── BaseExtractionPipeline
    └── ExtractionVlmPipeline
```

**Model Organization:**
```
Models/
├── BasePageModel (layout, ocr, table, assemble)
├── BaseVlmModel (vlm processing)
├── GenericEnrichmentModel (post-processing)
└── Factories (pluggable instantiation)
```

---

### 9.2 Data Flow

**High-level:**
```
Input -> InputDocument -> Pipeline -> Pages -> Assembly -> Document -> Enrichment -> Result
```

**Detailed Flow:**
```
1. Path/Stream/URL
   ↓
2. InputDocument (with backend)
   ↓
3. Pipeline.execute(InputDocument)
   ↓
4. Build Phase: Pages with predictions
   ↓
5. Assembly Phase: AssembledUnit
   ↓
6. ReadingOrder: DoclingDocument
   ↓
7. Enrichment: Enhanced DoclingDocument
   ↓
8. ConversionResult (document, confidence, timings)
```

---

### 9.3 Component Relationships

**Converter -> Pipeline:**
```python
converter = DocumentConverter(format_options={
    InputFormat.PDF: FormatOption(
        backend=PdfDocumentBackend,
        pipeline=StandardPdfPipeline,
        pipeline_options=PdfPipelineOptions(...)
    )
})
```

**Pipeline -> Backend:**
```python
pipeline = StandardPdfPipeline(pipeline_options)
in_doc = InputDocument(file=path, backend=PdfDocumentBackend(...))
result = pipeline.execute(in_doc)
```

**Pipeline -> Models:**
```python
layout_model = LayoutModel(layout_options)
ocr_model = ocr_factory.get_model(ocr_options)
table_model = TableStructureModel(table_options)
# Models called in pipeline stages
```

**Models -> Document:**
```python
# Models produce predictions
page.predictions.layout = layout_predictions
page.predictions.tablestructure = table_predictions

# Assembly creates document
assembled_unit = page_assemble_model(page)
document = reading_order_model(assembled_unit)
```

---

### 9.4 External Dependencies

**Key Libraries:**
- `docling-core`: Core data structures, chunking
- `docling-ibm-models`: AI models (Layout, Table, ReadingOrder)
- `docling-parse`: PDF parsing backend
- `transformers`: HuggingFace models
- `mlx`: Apple Silicon acceleration
- `vllm`: High-throughput VLM serving
- `whisper`: Audio transcription
- `tesseract`, `easyocr`, `rapidocr`: OCR engines
- `pypdfium2`: PDF backend
- `beautifulsoup4`: HTML parsing
- `markdown-it-py`: Markdown parsing
- `pydantic`: Data validation
- `PIL`: Image processing

---

## 10. Output Formats

### 10.1 DoclingDocument Structure

**Core Components:**

```python
class DoclingDocument:
    name: str                           # Document name
    origin: DocumentOrigin              # Filename, MIME, hash
    pages: List[PageItem]               # Page metadata
    furniture: GroupItem                # Headers, footers
    body: GroupItem                     # Main content
```

**Content Types:**

1. **TextItem:** Paragraphs, headers, captions, footnotes
2. **ListItem:** Bullet/numbered lists with markers
3. **Table:** Structured tables with cells
4. **Picture:** Images with optional captions
5. **SectionHeaderItem:** Section headers
6. **CodeItem:** Code blocks with language
7. **FormulaItem:** Mathematical formulas (text + LaTeX)
8. **GroupItem:** Lists, forms, key-value areas
9. **KeyValueItem:** Form fields

**Provenance:**
```python
class ProvenanceItem:
    page_no: int                        # 1-indexed page number
    bbox: BoundingBox                   # (x0, y0, x1, y1)
    charspan: Tuple[int, int]           # Character span in page text
```

**Metadata:**
```python
class DocumentOrigin:
    filename: str                       # Original filename
    mimetype: str                       # MIME type
    binary_hash: str                    # SHA256 hash
```

---

### 10.2 Export Formats

#### 1. Markdown Export

```python
markdown = result.document.export_to_markdown()
```

**Features:**
- Hierarchical headers
- Tables in markdown format
- Code blocks with syntax highlighting
- Math formulas in LaTeX
- Image references

**Example:**
```markdown
## Section Header

This is a paragraph with **bold** and *italic* text.

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

```python
print("Hello, World!")
```

Formula: $E = mc^2$
```

#### 2. JSON Export

```python
json_str = result.document.export_to_json()
```

**Features:**
- Complete document structure
- Provenance information
- Bounding boxes
- Nested elements
- Metadata

#### 3. HTML Export

```python
html = result.document.export_to_html()
```

**Features:**
- Semantic HTML5
- CSS classes for styling
- Table structure preservation
- Image embedding (base64 or links)

#### 4. Text Export

```python
text = result.document.export_to_text()
```

**Features:**
- Plain text extraction
- Reading order preserved
- Minimal formatting

#### 5. DocTags Export

```python
doctags = result.document.export_to_doctags()
```

**Features:**
- Structured tags with positions
- Bounding box information
- Element type labels
- Used for training/evaluation

**Example:**
```
<TEXT bbox="10,20,100,30">This is text content</TEXT>
<TABLE bbox="10,40,200,150">...</TABLE>
```

---

### 10.3 Chunking Output

#### HierarchicalChunker Output

```python
from docling.chunking import HierarchicalChunker

chunker = HierarchicalChunker(max_chunk_size=512, overlap=50)
chunks = list(chunker.chunk(result.document))
```

**Chunk Structure:**
```python
class DocChunk:
    text: str                           # Chunk text
    metadata: Dict                      # Page number, section, etc.
    doc_items: List[NodeItem]           # Source document items
    level: int                          # Hierarchy level
```

**Metadata:**
```python
{
    "page_no": 1,
    "section": "Introduction",
    "doc_items": ["paragraph_1", "paragraph_2"],
    "chunk_id": "chunk_001"
}
```

**Use Cases:**
- RAG (Retrieval Augmented Generation)
- Vector database ingestion
- Semantic search
- Embedding generation

---

## Summary

Docling is a comprehensive document understanding framework featuring:

**Architecture:**
- Modular, pipeline-based design
- Pluggable backends and models
- Multi-threaded processing for performance
- Extensible via plugin system

**Capabilities:**
- 20+ document format support
- Advanced PDF understanding with AI models
- Vision-Language Model integration
- Structured data extraction
- Audio/video transcription

**Processing:**
- Layout detection and classification
- Table structure extraction
- OCR for scanned documents
- Reading order determination
- Document enrichment (code, formulas, picture descriptions)

**Output:**
- Rich `DoclingDocument` with hierarchy
- Multiple export formats (MD, JSON, HTML, DocTags)
- Chunking for RAG applications
- Quality metrics (confidence scores)

**Production-Ready:**
- Thread-safe execution
- Error handling and recovery
- Performance tuning options
- Batch processing support
- Quality monitoring

The framework excels at converting unstructured documents into structured, searchable, and analyzable formats suitable for RAG, search engines, document analytics, and other AI applications.
