# ECEasy Image Support Documentation

## Overview

The updated ECEasy FAISS ingestion system now supports:
1. **Image cataloging** from `./ECEknowledge` (PNG, JPG, JPEG)
2. **Image metadata extraction** from file paths and names
3. **Image manifest generation** (JSON) for backend queries
4. **Backend image retrieval** module for searching and suggesting images in chat responses

## File Structure

- **`ingest_FAISS.py`** - Main ingestion script that:
  - Loads all PDFs, DOCXs, TXTs from `./ECEknowledge`
  - Scans for and catalogs all images
  - Extracts metadata (course code, department, doc type)
  - Builds FAISS vector index
  - Generates `image_manifest.json` in the FAISS index folder

- **`image_retrieval.py`** - Backend helper module for:
  - Loading image manifests
  - Searching images by course code, department, keywords
  - Suggesting relevant images for chatbot responses
  - Getting image file paths for display

## Running Ingestion

```bash
python ingest_FAISS.py
```

Output:
- `faiss_index_bge-small-en-v1.5/` (or your configured model name)
  - `index.faiss` - Vector database
  - `index.pkl` - Metadata
  - **`image_manifest.json`** - Image catalog (NEW)

## Image Naming Conventions

For best metadata extraction, name images descriptively:

### Good examples:
```
ELEC_curriculum_25-26.png           → dept=ELEC, type=general, desc="ELEC curriculum 25 26"
Sample_ELEC_Study_Pattern.png       → dept=ELEC, type=general, desc="Sample ELEC Study Pattern"
COMP2011_Prerequisites.png          → code=COMP2011, dept=COMP
Common_Core_Course.png              → type=course_syllabus, desc="Common Core Course"
course syllabus/ELEC/ELEC1100_flowchart.png → code=ELEC1100, dept=ELEC, type=course_syllabus
```

### Image metadata extracted:
- **`source_relpath`** - Path relative to ECEknowledge (e.g., `"ELEC_curriculum_25-26.png"`)
- **`source_name`** - Filename (e.g., `"ELEC_curriculum_25-26.png"`)
- **`description`** - Human-readable from filename (e.g., `"ELEC curriculum 25 26"`)
- **`course_code`** - If detected in path/name (e.g., `"ELEC1100"`)
- **`department`** - Extracted from course code or path (e.g., `"ELEC"`)
- **`doc_type`** - Inferred from folder structure:
  - `"course_syllabus"` if in `course syllabus/` folder
  - `"program_requirement"` if in `program requirement/` folder
  - `"faq"` if filename contains "faq"
  - `"general"` otherwise
- **`file_size_bytes`** - Image file size

## Backend Integration

### Quick Start

```python
from image_retrieval import ImageRetriever, suggest_images_for_response

# Initialize retriever (auto-finds manifest from current FAISS index)
retriever = ImageRetriever()

# Search by course code
images = retriever.search_by_course_code("ELEC1100")

# Search by department
images = retriever.search_by_department("ELEC")

# Search by keywords
images = retriever.search_by_keywords(["study", "pattern"])

# Fuzzy search by query
images = retriever.search_by_query("ELEC curriculum requirements")

# Get image file path
image_info = images[0]
image_path = retriever.get_image_path(image_info)
print(f"Display: {image_path}")  # ECEknowledge/ELEC_curriculum_25-26.png

# Get all images
all_images = retriever.get_all_images()
```

### Integrate with Chatbot Response

```python
from image_retrieval import suggest_images_for_response

query = "What does the ELEC curriculum look like?"
response = "The ELEC curriculum consists of..."

# Suggest relevant images for this response
images = suggest_images_for_response(query, response, retriever)

# Return images to frontend
response_with_images = {
    "text": response,
    "suggested_images": [
        {
            "path": "ECEknowledge/" + img["source_relpath"],
            "description": img["description"],
            "doc_type": img["doc_type"],
        }
        for img in images
    ]
}
```

### Search Methods

#### By Course Code
```python
images = retriever.search_by_course_code("ELEC1100")
# Returns all images with course_code="ELEC1100"
```

#### By Department
```python
images = retriever.search_by_department("ELEC")
# Returns all images with department="ELEC"
```

#### By Document Type
```python
images = retriever.search_by_doc_type("course_syllabus")
# Returns: "course_syllabus", "program_requirement", "faq", "general"
```

#### By Keywords
```python
images = retriever.search_by_keywords(["study", "pattern"], field="description")
# Searches in specified field (default: "description")
# Available fields: "description", "source_name", "source_relpath"
```

#### Fuzzy Query Search (Best for User Queries)
```python
images = retriever.search_by_query("ELEC study requirements curriculum")
# Searches across all metadata fields with relevance scoring
# Returns results sorted by relevance score
```

#### By Exact Path
```python
image = retriever.get_image_by_relpath("ELEC_curriculum_25-26.png")
# Returns single image or None
```

## Image Manifest JSON Structure

Example `image_manifest.json`:

```json
[
  {
    "source_relpath": "ELEC_curriculum_25-26.png",
    "source_name": "ELEC_curriculum_25-26.png",
    "source_stem": "ELEC_curriculum_25-26",
    "file_size_bytes": 1554161,
    "doc_type": "general",
    "department": "",
    "course_code": "",
    "description": "ELEC curriculum 25 26"
  },
  {
    "source_relpath": "course syllabus/common core courese/Common_Core_Course.png",
    "source_name": "Common_Core_Course.png",
    "source_stem": "Common_Core_Course",
    "file_size_bytes": 759445,
    "doc_type": "course_syllabus",
    "department": "",
    "course_code": "",
    "description": "Common Core Course"
  }
]
```

## Frontend Integration Example

When the chatbot returns a response with suggested images:

```javascript
// Assuming response from backend includes:
const response = {
  text: "The ELEC curriculum...",
  suggested_images: [
    {
      path: "ECEknowledge/ELEC_curriculum_25-26.png",
      description: "ELEC curriculum 25 26",
      doc_type: "general"
    }
  ]
};

// Display images
response.suggested_images.forEach(img => {
  const imgElement = document.createElement("img");
  imgElement.src = img.path;
  imgElement.alt = img.description;
  imgElement.title = `[${img.doc_type}] ${img.description}`;
  chatContainer.appendChild(imgElement);
});
```

## Best Practices

1. **Descriptive Filenames** - Use underscores/hyphens to separate words
   - ✅ `ELEC_Study_Pattern_2025.png`
   - ❌ `pattern.png` or `pic123.png`

2. **Organized Folder Structure** - Use meaningful subdirectories
   - ✅ `course syllabus/ELEC/ELEC1100_flowchart.png`
   - ✅ `FYP_FYT_Co-Op/progress_report_template.png`
   - ❌ All images in root folder

3. **Image Format** - Supported: PNG, JPG, JPEG
   - Keep file sizes < 2 MB for optimal frontend performance

4. **Consistent Naming** - Include relevant identifiers
   - Course code when applicable: `ELEC1100_...`
   - Department for general images: `ELEC_...`
   - Meaningful keywords: `Study_Pattern`, `Curriculum`, `Requirements`

5. **Re-ingestion** - After adding new images:
   ```bash
   python ingest_FAISS.py
   ```
   This will regenerate the `image_manifest.json`

## Troubleshooting

### Image not appearing in manifest

1. Check file format (must be `.png`, `.jpg`, or `.jpeg`)
2. Check file location (must be under `ECEknowledge/` recursively)
3. Re-run `python ingest_FAISS.py`
4. Check `image_manifest.json` exists in FAISS index folder

### Search returning no results

1. Use `retriever.get_all_images()` to verify images are loaded
2. Check metadata extraction - image names should match search criteria
3. Try different search methods (keywords vs. query vs. by_department)

### Image path not accessible

1. Verify `source_relpath` is correct: `ECEknowledge/{source_relpath}`
2. Check file permissions
3. Use `retriever.get_image_path(img_metadata)` to get correct Path object

## Performance Notes

- Image metadata is lightweight (~1 KB per image in JSON)
- Manifest loads in O(1) time
- Searches are O(n) where n = number of images (typically < 100)
- No GPU required for image metadata operations
- Images are not embedded or indexed by FAISS (only text is embedded)

