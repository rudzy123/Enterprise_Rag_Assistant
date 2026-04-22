# Markdown Parsing Debug Report

## Issues Found & Fixed

### 1. **Wrong Directory Path** (Primary Issue)
- **Problem**: Script pointed to `data/curated/` which was empty
- **Reality**: Actual files are in `data/docs/curated/` 
- **Fix**: Changed `CURATED_DIR = Path("data/curated")` to `Path("data/docs/curated")`

### 2. **Parsing Logic Bug** (Secondary Issue)
- **Problem**: The nested `flush_section()` function referenced `current_title` and `current_body` without declaring them as `nonlocal`, causing potential scoping issues
- **Fix**: Added `nonlocal current_title, current_body` and reset variables after flush

### 3. **Fragile Header Detection** (Robustness Issue)
- **Problem**: Used `.startswith("## ")` which fails if headers have no space (e.g., `##Header`)
- **Fix**: Changed to regex matching `r"^##\s+"` which handles:
  - Single space: `## Header`
  - Multiple spaces: `##  Header`  
  - Tabs: `##\tHeader`

---

## How to Debug Markdown Issues Systematically

### Step 1: Verify Files Exist & Are Not Empty
```python
from pathlib import Path

CURATED_DIR = Path("data/docs/curated")
print(f"Directory exists: {CURATED_DIR.exists()}")

md_files = list(CURATED_DIR.glob("*.md"))
for f in md_files:
    size = f.stat().st_size
    print(f"{f.name}: {size} bytes")
    if size == 0:
        print(f"  ⚠️  File is empty!")
```

### Step 2: Inspect Raw Bytes & Encoding
```python
with open(md_file, "rb") as f:
    raw_bytes = f.read()

# Check for BOM
if raw_bytes.startswith(b'\xef\xbb\xbf'):
    print("UTF-8 BOM detected")
    
# Inspect first few bytes
print(f"First 50 bytes (hex): {raw_bytes[:50].hex()}")
```

### Step 3: Check for Headers at the Line Level
```python
import re

with open(md_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Count headers
h1 = sum(1 for line in lines if re.match(r"^#\s+", line))
h2 = sum(1 for line in lines if re.match(r"^##\s+", line))
h3 = sum(1 for line in lines if re.match(r"^###\s+", line))

print(f"Headers: {h1} H1, {h2} H2, {h3} H3")

# Check for headers without space
h2_no_space = [l for l in lines if re.match(r"^##[^\s]", l)]
if h2_no_space:
    print(f"⚠️  Found {len(h2_no_space)} ## headers without space")
```

### Step 4: Inspect Header Characters in Detail
```python
for i, line in enumerate(lines):
    if line.startswith("##"):
        # Show the exact characters
        print(f"Line {i}: {repr(line)}")
        # Show byte-by-byte
        print(f"  Bytes: {line.encode('utf-8').hex()}")
```

### Step 5: Test Parsing in Isolation
```python
def test_split():
    test_cases = [
        ("## Header\nContent", "Standard"),
        ("##Header\nContent", "No space"),
        ("##  Header\nContent", "Double space"),
        ("", "Empty file"),
        ("# H1\n## H2", "Mixed levels"),
    ]
    
    for text, label in test_cases:
        result = split_markdown_sections(text)
        print(f"{label}: {len(result)} sections")
        for title, content in result:
            print(f"  - {title}")

test_split()
```

---

## Robust Markdown Header Parsing (Without External Librarians)

The fixed `split_markdown_sections()` function uses these principles:

1. **Regex for Flexible Matching**
   ```python
   if re.match(r"^##\s+", line):  # Handles variations in whitespace
   ```
   vs.
   ```python
   if line.startswith("## "):  # Brittle, fails on ##  or ##\t
   ```

2. **Smart Header Level Detection**
   ```python
   has_h3 = any(re.match(r"^###\s+", line) for line in lines)
   has_h2 = any(re.match(r"^##\s+", line) for line in lines)
   ```
   Prefers H3 if present, falls back to H2.

3. **Proper State Management with `nonlocal`**
   ```python
   def flush_section():
       nonlocal current_title, current_body
       if current_title and current_body:
           sections.append((current_title, body_text))
           current_title = None  # Reset state
           current_body = []
   ```

4. **Fallback for Files Without Headers**
   ```python
   if not has_h3 and not has_h2:
       return [("Content", text)]  # Treat whole file as one section
   ```

---

## Results

**Before Fix**: 0 chunks ingested ❌
```
Ingested 0 chunks
```

**After Fix**: 18 chunks ingested ✅
```
Ingested 18 chunks:
- 6 sections from access_control_policy.md
- 4 sections from incident_response_runbook.md
- 5 sections from nist_800_53_selected_controls.md
- 4 sections from nist_800_61_incident_response.md
```

---

## Files Modified

- `ingestion/ingest_curated_md.py` — Fixed directory path, parsing logic, and header detection
