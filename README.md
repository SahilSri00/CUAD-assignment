# Legal Contract Analysis Pipeline

An LLM-powered pipeline that analyzes legal contracts from the [CUAD dataset](https://www.atticusprojectai.org/cuad), extracts key clauses (Termination, Confidentiality, Liability), and generates concise summaries.

## Architecture & Flow Diagram

```mermaid
flowchart TD
    A["CUAD Dataset\n(Zenodo)"] -->|Download & Extract| B["Data Loader\n(downloader.py)"]
    B -->|Select 50 contracts| C["PDF Parser\n(parser.py)"]
    C -->|Extract text\nvia PyMuPDF| D{Text > 10K tokens?}
    D -->|No| E["Single-pass\nLLM Processing"]
    D -->|Yes| F["Chunk with\noverlap"]
    F --> G["Multi-chunk\nLLM Processing"]
    G --> H["Merge & Deduplicate\nChunk Results"]
    E --> I["Clause Extraction\n(extractor.py)"]
    H --> I
    I -->|Few-shot prompting\nwith JSON output| J["Extracted Clauses\n• Termination\n• Confidentiality\n• Liability"]
    I --> K["Contract Summary\n100-150 words"]
    J --> L["Output Files\nCSV + JSON"]
    K --> L
    L --> M["Evaluation\n(evaluator.py)"]
    M -->|Jaccard similarity\nvs CUAD ground truth| N["Evaluation Report"]
    L --> O["Semantic Search Index\n(search.py)"]
    O -->|OpenAI Embeddings\n+ Cosine Similarity| P["Query Interface"]

    style A fill:#4A90D9,color:#fff
    style I fill:#E67E22,color:#fff
    style L fill:#27AE60,color:#fff
    style O fill:#8E44AD,color:#fff
    style N fill:#C0392B,color:#fff
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE FLOW (ASCII)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐              │
│  │ Download  │───>│  Parse   │───>│   Normalize  │              │
│  │ CUAD v1   │    │  PDFs    │    │   + Chunk    │              │
│  └──────────┘    └──────────┘    └──────┬───────┘              │
│                                         │                       │
│                                         ▼                       │
│                              ┌─────────────────────┐           │
│                              │   LLM Processing     │           │
│                              │  (gpt-4o-mini)       │           │
│                              │                       │           │
│                              │  ┌─────────────────┐ │           │
│                              │  │ Clause Extraction│ │           │
│                              │  │ (few-shot)       │ │           │
│                              │  └─────────────────┘ │           │
│                              │  ┌─────────────────┐ │           │
│                              │  │ Summarization    │ │           │
│                              │  │ (100-150 words)  │ │           │
│                              │  └─────────────────┘ │           │
│                              └──────────┬──────────┘           │
│                                         │                       │
│                         ┌───────────────┼───────────────┐      │
│                         ▼               ▼               ▼      │
│                  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│                  │  CSV/JSON  │  │ Evaluation  │  │ Semantic │  │
│                  │  Output    │  │  Report     │  │ Search   │  │
│                  └────────────┘  └────────────┘  └──────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites
- Python 3.9+
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd contract-analysis-pipeline

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# On Windows:
set OPENAI_API_KEY=your-api-key-here
```

## Usage

### Run the Full Pipeline

```bash
# Process 50 contracts (default)
python main.py

# Process a smaller subset for testing
python main.py --contracts 10

# Use a different model
python main.py --model gpt-4o

# Enable verbose logging
python main.py --verbose
```

### Semantic Search (Bonus)

After running the pipeline, search over extracted clauses:

```bash
python main.py --search "limitation of liability capping"
python main.py --search "termination for convenience 30 days notice"
python main.py --search "non-disclosure of confidential information"
```

## Approach

### 1. Data Loading & Preprocessing
- The CUAD v1 dataset (510 contracts) is automatically downloaded from Zenodo.
- A deterministic subset of 50 contracts is selected using evenly-spaced sampling across the alphabetically sorted list, ensuring diversity across contract types.
- PDF text is extracted using **PyMuPDF** (fitz), with a fallback to pre-extracted TXT files.
- Text normalization includes: removing SEC boilerplate/confidentiality legends, fixing encoding artifacts, collapsing whitespace, and removing page numbers.

### 2. Chunking Strategy
- Contracts exceeding ~10K tokens are split into overlapping chunks at paragraph boundaries.
- Each chunk has ~500 tokens of overlap with the previous chunk to avoid cutting clauses mid-sentence.
- For multi-chunk contracts, each chunk is processed separately and results are merged via a dedicated LLM consolidation call.

### 3. Prompt Engineering
- **Clause Extraction**: Uses a structured system prompt defining the three clause types with clear descriptions. Includes **two few-shot examples** (one with all three clauses present, one with a missing confidentiality clause) to calibrate the model's extraction behavior. Outputs structured JSON via `response_format={"type": "json_object"}`.
- **Summarization**: Separate prompt with explicit 100–150 word constraint, covering purpose, obligations, and risks. Includes a programmatic word-count validation with automatic retry if outside the target range.
- **Model**: Uses `gpt-4o-mini` by default for its strong performance-to-cost ratio. Configurable via `--model` flag.

### 4. Evaluation
- Compares LLM-extracted Termination and Liability clauses against CUAD's human annotations using:
  - **Jaccard similarity** on token sets (measures text overlap)
  - **Detection accuracy** (did the model correctly identify presence/absence of a clause?)
- Confidentiality clauses are not evaluated against ground truth since CUAD does not contain a dedicated Confidentiality category.

### 5. Semantic Search (Bonus)
- All extracted clauses and summaries are embedded using OpenAI's `text-embedding-3-small` model.
- Search queries are embedded and compared via cosine similarity.
- The index is persisted to disk for fast reuse.

## Output Format

### CSV (`output/contracts_analysis.csv`)
| Column | Description |
|--------|-------------|
| `contract_id` | Contract filename (stem) |
| `summary` | 100–150 word contract summary |
| `termination_clause` | Extracted termination conditions |
| `confidentiality_clause` | Extracted confidentiality obligations |
| `liability_clause` | Extracted liability limitations |

### JSON (`output/contracts_analysis.json`)
Same structure as CSV, formatted as a list of objects.

## Project Structure

```
├── main.py                  # CLI entrypoint
├── requirements.txt         # Dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── core/
│   ├── __init__.py          # Package init
│   ├── downloader.py        # Dataset download & subset selection
│   ├── parser.py            # PDF parsing & text normalization
│   ├── extractor.py         # LLM clause extraction & summarization
│   ├── search.py            # Semantic search with embeddings
│   └── evaluator.py         # Ground truth evaluation
└── output/                  # Generated at runtime
    ├── contracts_analysis.csv
    ├── contracts_analysis.json
    ├── evaluation_metrics.json
    ├── search_metadata.json
    └── search_embeddings.npy
```

## Limitations & Future Work
- **Confidentiality evaluation**: CUAD lacks a dedicated Confidentiality category, so ground-truth evaluation is limited to Termination and Liability clauses.
- **PDF quality**: Some CUAD PDFs have OCR artifacts or complex table layouts that may affect extraction quality.
- **Model comparison**: The pipeline supports swapping models via `--model` flag (e.g., `gpt-4o` vs `gpt-4o-mini`) for comparative analysis.
- **Scalability**: For production use, consider async API calls, caching, and a vector database (e.g., FAISS, Pinecone) for the search index.

## License

This project uses the CUAD dataset, which is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
