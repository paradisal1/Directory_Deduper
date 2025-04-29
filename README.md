# Duplicate File Finder

## Overview
The **Duplicate File Finder** is a Python-based application designed to scan directories recursively, identify duplicate files, and provide detailed insights into duplicate groups. It uses efficient file scanning and hashing techniques to compare file contents, ensuring accurate detection of duplicates while maintaining high performance. The application also features a user-friendly interface built with Streamlit, allowing users to interactively scan directories and analyze results.

---

## Features
- **Recursive Directory Scanning**: Efficiently scans directories and subdirectories to collect file information.
- **Duplicate Detection**: Identifies duplicate files by comparing their content using a three-step size comparison and hashing process.
- **Error Handling**: Gracefully handles permission errors and inaccessible files, logging them for review.
- **Streamlit UI**: Provides an intuitive interface for scanning directories, viewing results, and exporting duplicate file lists.
- **Export Results**: Allows users to export duplicate file details as a CSV file for further analysis.
- **Visualization**: Displays duplicate groups and their total sizes using bar charts for easy interpretation.

---

## How It Works

### 1. **Directory Scanning**
The application uses the `fast_scan_directory` function to traverse the directory tree iteratively (avoiding recursion to handle deeply nested structures). It collects file paths, sizes, and handles permission errors gracefully.

### 2. **Three-Step Deduplication/Hashing Process**
To optimize performance, the application employs a three-step deduplication/hashing process:
- **File Size Comparison**: Rapidly removes non-duplicates through file size comparisons.
- **First Chunk Hashing**: Quickly eliminates non-duplicates by hashing only the first chunk of each file.
- **Full File Hashing**: For files with matching first-chunk hashes, the entire file content is hashed to confirm duplicates.

### 3. **Duplicate Grouping**
Files with identical hashes are grouped together as duplicates. The application provides options to group duplicates by their hash or directory for better organization.

### 4. **Streamlit Interface**
The Streamlit-based UI allows users to:
- Input the directory path to scan.
- View scan summaries, including total files scanned, total size, and duplicate groups.
- Export duplicate file details as a CSV file.
- Visualize duplicate groups by their total size using bar charts.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

### Steps
1. Clone the repository:
```bash
git clone 
```

2. Run 
```bash
pip install uv
cd gmb_app
uv sync
```

2. (Alternatively) Run
```bash
cd gmb_app
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the Application
Run the application using the command:
```bash
cd gmb_app
streamlit run main.py
```

### 2. Input Directory
Enter the path of the directory you want to scan in the text input field.

### 3. Start Scanning
Click the **"Scan for duplicates"** button to begin the scan. The application will:
- Traverse the directory tree.
- Identify duplicate files.
- Handle permission errors gracefully.

### 4. View Results
- **Scan Summary**: Displays total files scanned, total size, duplicate groups found, and scan duration.
- **Duplicate Groups**: Lists duplicate files grouped by their hash or directory.
- **Visualization**: Shows a bar chart of duplicate groups by total size.

### 5. Export Results
Click the **"Export Duplicate File List"** button to download a CSV file containing details of duplicate files.

---

## Configuration

The application includes several configurable options in the `main.py` file:

| Option                          | Description                                                                 | Default Value       |
|----------------------------------|-----------------------------------------------------------------------------|---------------------|
| `USE_MMAP`                      | Use memory-mapped file access for hashing smaller files.                    | `True`              |
| `USE_CHUNKED_MMAP_FOR_LARGE_FILES` | Fallback to manual chunking for large files if memory-mapped access is too large. Not yet implemented | `True`              |
| `USE_MULTITHREADED_HASHING`     | Enable multi-core parallel hashing for improved performance.                | `True`              |
| `MMAP_THRESHOLD_BYTES`          | File size threshold (in bytes) for using memory-mapped access.              | `2 * 1024 ** 3` (2 GB) |
| `READ_BUFFER`                   | Chunk size (in bytes) for manual reads during hashing.                      | `8192` (8 KB)       |
| `CHUNK_SIZE`                    | Chunk size (in bytes) for initial hashing.                                  | `1024 * 1024` (1 MB)|
| `MAX_WORKERS`                   | Maximum number of workers for multithreaded hashing.                        | `os.cpu_count()`    |

## Output

### Scan Summary
- **Total Files Scanned**: Number of files scanned during the process.
- **Total Size Scanned**: Cumulative size of all scanned files.
- **Duplicate Groups Found**: Number of groups of duplicate files.
- **Scan Duration**: Time taken to complete the scan.

### Duplicate Groups
- Lists duplicate files grouped by their hash or directory.
- Provides file paths, sizes, and hash values for each duplicate.

### CSV Export
- Exports duplicate file details, including file paths, sizes, and hash values, as a CSV file.

---

## Visualization

The application generates a bar chart to visualize duplicate groups by their total size. This helps users quickly identify the largest groups of duplicate files and prioritize their cleanup efforts.

---

## Example

### Input
Directory to scan: `/path/to/directory`

### Output
- **Scan Summary**:
  - Total Files Scanned: 10,000
  - Total Size Scanned: 50 GB
  - Duplicate Groups Found: 200
  - Scan Duration: 30 seconds
- **Duplicate Groups**:
  - Group 1: `file1.txt`, `file2.txt` (Hash: `abc123`, Size: 5 MB)
  - Group 2: `image1.jpg`, `image2.jpg` (Hash: `def456`, Size: 10 MB)
- **CSV Export**: A CSV file containing details of duplicate files.

---

## Limitations
- **Large Files**: Scanning very large files may take longer, depending on system resources.
- **Permission Errors**: Files or directories with restricted access are skipped, which may affect results.
- **Symlinks**: The application does not follow symbolic links to avoid infinite loops and redundant scanning.