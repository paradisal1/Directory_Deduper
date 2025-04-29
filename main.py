import os
import mmap
import xxhash
import asyncio
# import hashlib
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed



#####################################################################################################
########################################## Config ###################################################
#####################################################################################################
# USE_MMAP = True  # Use memory-mapped file access, better for smaller files
# USE_MULTITHREADED_HASHING = True  # Use multi-core parallel hashing
# MMAP_THRESHOLD_BYTES = 2 * 1024 ** 3  # 2 GB threshold for mmap 
# READ_BUFFER = 8192  # Chunk size for manual reads
MAX_WORKERS = os.cpu_count()
# CHUNK_SIZE = 1024 * 1024  # 1 MB chunk size for initial hashing

## Try deduplicating live during scanning (not yet implemented, only useful for >50k files)
## Requires creation of separate functions for live and batch deduping
# DEDUPE_WHILE_SCANNING = True  

## Fallback to manual chunking if mmap too large (not yet implemented, defaults to chunked read based on file size)
## Used in the _read_full_hash function
# USE_CHUNKED_MMAP_FOR_LARGE_FILES = True  


executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

HASH_ALGORITHM = xxhash.xxh3_64
# HASH_ALGORITHM = hashlib.md5
HASH_ALGORITHM('THIS IS A TEST')

X = HASH_ALGORITHM().update()

#####################################################################################################
####################################### Utility Functions ###########################################
#####################################################################################################
async def gather_in_batches(tasks, total_files=None, base_batch_size=100, delay_between_batches=0.05):
    """
    Gather asyncio tasks in safe batches, dynamically adjusting batch size if total files are huge.

    Args:
        tasks (list): List of asyncio tasks.
        total_files (int): Total number of files scanned (optional).
        base_batch_size (int): Base number of concurrent tasks.
        delay_between_batches (float): Pause between batches (seconds).
    
    Returns:
        list: Combined results of all tasks.
    """
    if total_files:
        if total_files > 200_000:
            batch_size = max(25, base_batch_size // 4)
        elif total_files > 100_000:
            batch_size = max(50, base_batch_size // 2)
        elif total_files < 10_000:
            batch_size = min(500, base_batch_size * 5)
        else:
            batch_size = base_batch_size
    else:
        batch_size = base_batch_size

    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        results.extend(await asyncio.gather(*batch))
        await asyncio.sleep(delay_between_batches)  # smoothing network IO
    return results



#####################################################################################################
######################################## Utility Classes ############################################
#####################################################################################################
class StreamlitProgressBar:
    def __init__(self, total, description="Working..."):
        self.progress_bar = st.progress(0, text=description)
        self.total = total
        self.current = 0
        self.description = description

    def update(self, increment=1):
        self.current += increment
        pct_complete = min(self.current / self.total, 1.0)
        self.progress_bar.progress(pct_complete, text=f"{self.description} ({self.current}/{self.total})")

    def done(self):
        self.progress_bar.empty()



#####################################################################################################
######################################## Fast Directory Scanner #####################################
#####################################################################################################
def fast_scan_directory(directory: Path):
    """
    Recursively scans a directory to collect information about files and their sizes.
    This function traverses the given directory and its subdirectories (non-recursively using a stack)
    to gather details about all files, while avoiding symbolic links to prevent infinite loops.
    It also handles permission errors gracefully by recording paths that cannot be accessed.

    Args:
        directory (Path): The root directory to start scanning.

    Returns:
        tuple: A tuple containing:
            - file_infos (list of tuple): A list of tuples where each tuple contains:
                - Path: The file path.
                - int: The file size in bytes.
            - total_files (int): The total number of files found.
            - total_size (int): The cumulative size of all files in bytes.
            - permission_error_paths (list of Path): A list of paths that could not be accessed due to permission errors or other issues.
    """
    file_infos = []
    permission_error_paths = []
    total_files = 0
    total_size = 0

    stack = [directory]
    thread_executor = ThreadPoolExecutor(max_workers=8)  # 8 threads for scanning

    futures = []

    def process_dir(path):
        """
        Processes a directory and retrieves its files, subdirectories, and any errors encountered.

        Args:
            path (str or Path): The path to the directory to process.

        Returns:
            tuple: A tuple containing three elements:
                - local_files (list): A list of tuples where each tuple contains a Path object 
                  representing a file and its size in bytes.
                - local_dirs (list): A list of Path objects representing subdirectories within the directory.
                - local_errors (list): A list of Path objects representing files or directories that 
                  could not be accessed due to errors such as PermissionError, FileNotFoundError, or OSError.
        """
        local_files = []
        local_dirs = []
        local_errors = []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            local_dirs.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            size = entry.stat(follow_symlinks=False).st_size
                            local_files.append((Path(entry.path), size))
                    except (PermissionError, FileNotFoundError, OSError):
                        local_errors.append(Path(entry.path))
        except (PermissionError, FileNotFoundError, OSError):
            local_errors.append(path)
        return local_files, local_dirs, local_errors

    # Start processing initial directory
    while stack or futures:
        while stack and len(futures) < 32:  # Queue up to 32 outstanding scans
            next_dir = stack.pop()
            futures.append(thread_executor.submit(process_dir, next_dir))

        done, futures = list(), list(futures)  # rebuild list each round
        for future in as_completed(futures):
            local_files, local_dirs, local_errors = future.result()

            file_infos.extend(local_files)
            stack.extend(local_dirs)
            permission_error_paths.extend(local_errors)

            done.append(future)

        for f in done:
            futures.remove(f)

    thread_executor.shutdown(wait=True)

    total_files = len(file_infos)
    total_size = sum(size for _, size in file_infos)

    return file_infos, total_files, total_size, permission_error_paths

# fast_scan_directory(str(Path.home()))  # Test the function on the home directory



#####################################################################################################
######################################## Hashing Functions ##########################################
#####################################################################################################
async def hash_first_chunk(file_path: Path) -> str:
    """
    Asynchronously computes the hash of the first chunk of a file.

    This function reads the first chunk of the specified file and computes its hash
    using a separate thread to avoid blocking the event loop.

    Args:
        file_path (Path): The path to the file whose first chunk will be hashed.

    Returns:
        str: The computed hash of the first chunk of the file.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _read_chunk_hash, file_path)


def _read_chunk_hash(file_path: Path) -> str:
    """
    Computes the hash of a file chunk using a predefined hash algorithm.

    This function reads a chunk of the file specified by `file_path` and computes
    its hash using the hash algorithm defined by the global `HASH_ALGORITHM`.
    The size of the chunk to be read is determined by the global `CHUNK_SIZE`.

    Args:
        file_path (Path): The path to the file whose chunk hash is to be computed.

    Returns:
        str: The hexadecimal digest of the computed hash. If a `PermissionError` or
             `OSError` occurs during file access, the string "PERMISSION_ERROR" is returned.
    """
    try:
        with open(file_path, 'rb') as f:
            return HASH_ALGORITHM(f.read(CHUNK_SIZE)).hexdigest()
    except (PermissionError, OSError):
        return "PERMISSION_ERROR"
    

async def hash_full(file_path: Path) -> str:
    """
    Asynchronously computes the hash of the entire contents of a file.

    Args:
        file_path (Path): The path to the file to be hashed.

    Returns:
        str: The computed hash of the file as a string.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _read_full_hash, file_path)


def _read_full_hash(file_path: Path) -> str:
    """
    Computes the full hash of a file using the specified hash algorithm.

    This function reads the entire content of the file at the given path and 
    computes its hash. It supports memory-mapped file access for improved 
    performance on smaller files, based on the `USE_MMAP` and `MMAP_THRESHOLD_BYTES` 
    configuration.

    Args:
        file_path (Path): The path to the file whose hash is to be computed.

    Returns:
        str: The hexadecimal digest of the file's hash. If a `PermissionError` 
        or `OSError` occurs during file access, the string "PERMISSION_ERROR" 
        is returned instead.
    """
    try:
        file_size = file_path.stat().st_size
        hasher = HASH_ALGORITHM()
        with open(file_path, 'rb') as f:
            # mmap = memory-mapped files, which essentially lazily loads the file contents
            # mmap is faster for smaller files, but for larger files we read in chunks
            # mmap is not always faster for larger files, so we use a threshold
            if USE_MMAP and file_size < MMAP_THRESHOLD_BYTES:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm: # fileno refers to the file descriptor of the open file
                    hasher.update(mm)
            else: # for large files
                while chunk := f.read(READ_BUFFER):
                    hasher.update(memoryview(chunk)) # memoryview allows us to avoid copying the data
        return hasher.hexdigest()
    except (PermissionError, OSError):
        return "PERMISSION_ERROR"
    

async def hash_files_batch(files):
    """
    Compute the hashes of a batch of files asynchronously.

    This function supports two modes of operation:
    1. Multithreaded hashing using a process pool for improved performance on large batches.
    2. Sequential asynchronous hashing using asyncio tasks.

    Args:
        files (list[str]): A list of file paths to compute hashes for.

    Returns:
        list[str]: A list of hash values corresponding to the input files.

    Notes:
        - The mode of operation is determined by the `USE_MULTITHREADED_HASHING` flag.
        - The number of workers for multithreaded hashing is controlled by the `MAX_WORKERS` constant.
    """
    if USE_MULTITHREADED_HASHING:
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            return await loop.run_in_executor(pool, _batch_hash_sync, files)
    else:
        tasks = [hash_full(file) for file in files]
        return await asyncio.gather(*tasks)
    

def _batch_hash_sync(files):
    """
    Computes the hash values for a batch of files synchronously.

    Args:
        files (list): A list of file paths to compute the hash values for.

    Returns:
        list: A list of hash values corresponding to the input files.
    """
    return [_read_full_hash(file) for file in files]



#####################################################################################################
######################################## Main Duplicate Finder ######################################
#####################################################################################################
async def find_duplicates(directory: Path):
    """
    Asynchronously finds duplicate files in a given directory based on their content.
    This function scans the specified directory to identify duplicate files by comparing
    their sizes, hashing their first chunks, and finally hashing their full content if necessary.
    It returns a dictionary of duplicate files grouped by their hash, along with some statistics
    about the scan.

    Args:
        directory (Path): The root directory to scan for duplicate files.

    Returns:
        tuple:
            - dupes_filtered (dict): A dictionary where keys are file hashes and values are lists
              of duplicate file paths.
            - total_files (int): The total number of files scanned.
            - total_size (int): The total size (in bytes) of all files scanned.
            - permission_error_paths (list): A list of file paths that could not be accessed due
              to permission errors.

    Notes:
        - The function uses a two-step hashing process after comparing file size to optimize performance:
            1. Hashes the first chunk of each file to quickly eliminate non-duplicates.
            2. Hashes the full content of files that have matching first-chunk hashes.
        - Files that cannot be accessed due to permission errors are skipped and logged
          in the `permission_error_paths` list.
    """
    file_infos, total_files, total_size, permission_error_paths = fast_scan_directory(directory)
    files_by_size = defaultdict(list)

    for path, size in file_infos:
        files_by_size[size].append(path)  # group by size logic

    size_groups = [group for group in files_by_size.values() if len(group) > 1]
    chunk_progress = StreamlitProgressBar(sum(len(g) for g in size_groups), description="Hashing First Chunks")
    chunk_hash_tasks = []

    for group in size_groups:
        for file in group:
            chunk_hash_tasks.append(hash_first_chunk(file))

    chunk_results = await gather_in_batches(chunk_hash_tasks, total_files=total_files)
    chunk_progress.done()

    files_by_chunk_hash = defaultdict(list)
    idx = 0
    for group in size_groups:
        for file in group:
            if chunk_results[idx] != "PERMISSION_ERROR":
                files_by_chunk_hash[(file.stat().st_size, chunk_results[idx])].append(file)
            idx += 1

    chunk_groups = [group for group in files_by_chunk_hash.values() if len(group) > 1]
    full_progress = StreamlitProgressBar(sum(len(g) for g in chunk_groups), description="Hashing Full Files")
    full_hash_tasks = []

    for group in chunk_groups:
        for file in group:
            full_hash_tasks.append(file)

    full_results = await hash_files_batch(full_hash_tasks)
    full_progress.done()

    duplicates = defaultdict(list)
    idx = 0
    for group in chunk_groups:
        for file in group:
            if full_results[idx] != "PERMISSION_ERROR":
                duplicates[full_results[idx]].append(file)
            idx += 1

    dupes_filtered = {k: v for k, v in duplicates.items() if len(v) > 1}
    return dupes_filtered, total_files, total_size, permission_error_paths



#####################################################################################################
######################################## Streamlit UI ###############################################
#####################################################################################################
st.set_page_config(layout="wide", page_title="Duplicate File Finder", page_icon="🔍")
st.title("Duplicate File Finder")

USE_MMAP = st.sidebar.checkbox("Use Memory-Mapped Files (mmap)", value=True)
# USE_CHUNKED_MMAP_FOR_LARGE_FILES = st.sidebar.checkbox("Fallback to Chunked Read for Large Files", value=True)
USE_MULTITHREADED_HASHING = st.sidebar.checkbox("Enable Multi-Core Hashing", value=True)
# DEDUPE_WHILE_SCANNING = st.sidebar.checkbox("Deduplicate While Scanning", value=True)

MMAP_THRESHOLD_GB = st.sidebar.slider("mmap Size Threshold (GB)", min_value=1, max_value=10, value=2)
MMAP_THRESHOLD_BYTES = MMAP_THRESHOLD_GB * 1024 ** 3

READ_BUFFER_KB = st.sidebar.slider("Read Buffer Size (KB)", min_value=4, max_value=1024, value=8)
READ_BUFFER = READ_BUFFER_KB * 1024

CHUNK_SIZE_MB = st.sidebar.slider("First Chunk Size (MB)", min_value=1, max_value=20, value=1)
CHUNK_SIZE = CHUNK_SIZE_MB * 1024 ** 2

MAX_WORKERS = st.sidebar.slider("Max Parallel Workers", min_value=1, max_value=os.cpu_count(), value=os.cpu_count())

directory_input = st.text_input("Enter the path to scan recursively:", value=str(Path.home()))
directory_input = directory_input.strip().strip('"')
directory = Path(directory_input)

col1, col2 = st.columns([0.2, 0.8])

with col1:
    group_option = st.selectbox("Group duplicates by:", ["Hash", "Directory"])
    scan_clicked = st.button("Scan for duplicates")

if scan_clicked:
    if not directory.exists():
        st.warning("The specified directory does not exist. Please enter a valid path.")
    elif not directory.is_dir():
        st.warning("The specified path is not a directory. Please enter a valid directory path.")
    else:
        with st.spinner("Scanning for duplicates..."):
            import time
            start_time = time.time()
            dupes, total_files, total_size, permission_error_paths = asyncio.run(find_duplicates(directory))
            end_time = time.time()

        dupe_size = sum([path.stat().st_size for paths in dupes.values() for path in paths])
        dupe_size_mb = dupe_size / (1024 ** 2)

        with col1:
            st.markdown(f"### Scan Summary")
            st.markdown(f"- **Total Files Scanned:** {total_files}")
            st.markdown(f"- **Total Size Scanned:** {total_size / (1024 ** 2):.2f} MB")
            st.markdown(f"- **Duplicate Groups Found:** {len(dupes)}")
            st.markdown(f"- **Scan Duration:** {end_time - start_time:.2f} seconds")
            st.markdown(f"- **Total Duplicate Size:** {dupe_size_mb:.2f} MB")

            if dupes:
                file_data = []
                for hash_val, paths in dupes.items():
                    for path in paths:
                        file_data.append({
                            "hash": hash_val,
                            "file_path": str(path),
                            "size_mb": path.stat().st_size / (1024 ** 2),
                        })
                file_df = pd.DataFrame(file_data)
                csv = file_df.to_csv(index=False).encode('utf-8')
                st.download_button("Export Duplicate File List", data=csv, file_name='duplicate_files.csv', mime='text/csv')

        with col2:
            if dupes:
                dupe_data = [{
                    "hash": hash_val,
                    "file_count": len(paths),
                    "total_size": sum(p.stat().st_size for p in paths) / (1024 ** 2),
                    "directory": str(paths[0].parent)
                } for hash_val, paths in dupes.items()]
                df = pd.DataFrame(dupe_data).sort_values(by="total_size", ascending=False)
                fig, ax = plt.subplots()
                fig.set_size_inches(10, max(2, len(df) * 0.4))  # dynamic height based on number of groups

                bars = ax.barh(range(len(df)), df["total_size"])

                # Remove y-axis ticks and labels
                ax.set_yticks([])
                ax.set_yticklabels([])

                ax.set_xlabel("Size (MB)")
                ax.set_title("Duplicate Groups by Total Size")

                for idx, (bar, hash_val) in enumerate(zip(bars, df["hash"])):
                    width = bar.get_width()
                    bar_height = bar.get_height()
                    center_y = bar.get_y() + bar_height / 2

                    if width > 4:  # If bar is wide enough, put hash inside
                        ax.text(width - 0.5, center_y, hash_val, va="center", ha="right", color="white", fontsize=8)
                    else:  # If bar is too small, put hash outside
                        ax.text(width + 0.5, center_y, hash_val, va="center", ha="left", color="black", fontsize=8)

                st.pyplot(fig)

        dup_group_expander = st.expander("View Duplicate Groups", expanded=False)

        if dupes:
            if group_option == "Hash":
                # Sort by total size of each hash group (largest first)
                sorted_dupes = sorted(
                    dupes.items(),
                    key=lambda x: sum(path.stat().st_size for path in x[1]),
                    reverse=True
                )
                for hash_val, paths in sorted_dupes:
                    dup_group_expander.markdown(f"**Hash:** `{hash_val}`")
                    for path in paths:
                        size_mb = path.stat().st_size / (1024 ** 2)
                        dup_group_expander.code(f"{path} ({size_mb:.2f} MB)", language='text')

            elif group_option == "Directory":
                grouped_by_dir = defaultdict(list)
                for paths in dupes.values():
                    for path in paths:
                        grouped_by_dir[str(path.parent)].append(path)

                # Sort by total size of each directory group (largest first)
                sorted_dirs = sorted(
                    grouped_by_dir.items(),
                    key=lambda x: sum(path.stat().st_size for path in x[1]),
                    reverse=True
                )
                for dir_path, paths in sorted_dirs:
                    dup_group_expander.markdown(f"**Directory:** `{dir_path}`")
                    for path in paths:
                        dup_group_expander.code(str(path), language='text')
        else:
            dup_group_expander.markdown("No duplicate files found.")

        if permission_error_paths:
            perm_expander = st.expander("Permission Errors", expanded=False)
            for path in permission_error_paths:
                perm_expander.code(str(path), language='text')
        else:
            st.success("No permission errors encountered during the scan.")
            
