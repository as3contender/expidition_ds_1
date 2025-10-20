#!/usr/bin/env python3
import boto3
import os
import argparse
import urllib3
import time
import threading
from pathlib import Path
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# S3 Configuration
ENDPOINT_URL = 'https://s3.ru-1.storage.selcloud.ru:443'
BUCKET_NAME = 'public-expedition'
LOCAL_DIR = './'

# Initialize S3 client with SSL verification disabled and optimized settings
s3_client = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id='df192b56d640449c91b0e53e64b83e6b', 
    aws_secret_access_key='e0710cf8aca54b2386de2625c69421b4', 
    verify=False,  # Disable SSL verification
    config=Config(
        signature_version='s3v4',
        max_pool_connections=50,  # Increase connection pool
        retries={'max_attempts': 3}
    )
)

# Global counters for progress tracking
download_stats = {
    'completed': 0,
    'total': 0,
    'downloaded_bytes': 0,
    'total_bytes': 0,
    'start_time': None,
    'lock': threading.Lock()
}


def format_bytes(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def update_progress(file_size):
    """Update download progress"""
    with download_stats['lock']:
        download_stats['completed'] += 1
        download_stats['downloaded_bytes'] += file_size
        
        elapsed = time.time() - download_stats['start_time']
        speed = download_stats['downloaded_bytes'] / elapsed if elapsed > 0 else 0
        progress = (download_stats['completed'] / download_stats['total']) * 100
        
        print(f"\r  Progress: [{download_stats['completed']}/{download_stats['total']}] "
              f"{progress:.1f}% | {format_bytes(download_stats['downloaded_bytes'])}/{format_bytes(download_stats['total_bytes'])} | "
              f"Speed: {format_bytes(speed)}/s", end='', flush=True)


def download_file(bucket, key, local_file, file_size):
    """Download a single file"""
    try:
        # Create directory if needed
        local_file_dir = os.path.dirname(local_file)
        Path(local_file_dir).mkdir(parents=True, exist_ok=True)
        
        # Skip if file exists and has correct size
        if os.path.exists(local_file) and os.path.getsize(local_file) == file_size:
            update_progress(file_size)
            return True
        
        # Download file
        s3_client.download_file(bucket, key, local_file)
        
        # Verify file size
        if os.path.getsize(local_file) != file_size:
            print(f"\nWarning: Size mismatch for {key}")
            return False
            
        update_progress(file_size)
        return True
        
    except Exception as e:
        print(f"\nError downloading {key}: {e}")
        return False


def download_folder_parallel(bucket, prefix, local_dir, max_workers=20):
    """Download all files from S3 folder using parallel downloads"""
    print(f"Scanning files in: {prefix}")
    
    # Get file list
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    files_to_download = []
    total_size = 0
    
    for page in pages:
        if 'Contents' not in page:
            print(f"No files found in {prefix}")
            return
            
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('/'):
                local_file = os.path.join(local_dir, key)
                files_to_download.append((key, local_file, obj['Size']))
                total_size += obj['Size']
    
    if not files_to_download:
        print(f"No files to download from {prefix}")
        return
    
    # Initialize progress tracking
    download_stats['total'] = len(files_to_download)
    download_stats['completed'] = 0
    download_stats['downloaded_bytes'] = 0
    download_stats['total_bytes'] = total_size
    download_stats['start_time'] = time.time()
    
    print(f"Downloading {len(files_to_download)} files ({format_bytes(total_size)}) using {max_workers} threads...")
    
    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(download_file, bucket, key, local_file, file_size): key
            for key, local_file, file_size in files_to_download
        }
        
        failed_files = []
        for future in as_completed(future_to_file):
            key = future_to_file[future]
            try:
                success = future.result()
                if not success:
                    failed_files.append(key)
            except Exception as e:
                print(f"\nUnexpected error with {key}: {e}")
                failed_files.append(key)
    
    print()  # New line after progress
    
    if failed_files:
        print(f"Failed to download {len(failed_files)} files:")
        for key in failed_files[:10]:  # Show first 10
            print(f"  - {key}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    else:
        print("✓ All files downloaded successfully!")


def verify_downloads(bucket, prefix, local_dir):
    """Verify downloaded files by comparing sizes"""
    print(f"\nVerifying files in: {prefix}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    total_files = 0
    missing_files = 0
    size_mismatches = 0
    correct_files = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
                
            total_files += 1
            remote_size = obj['Size']
            local_file = os.path.join(local_dir, key)
            
            if not os.path.exists(local_file):
                missing_files += 1
                print(f"Missing: {key}")
            else:
                local_size = os.path.getsize(local_file)
                if local_size != remote_size:
                    size_mismatches += 1
                    print(f"Size mismatch: {key} (local: {format_bytes(local_size)}, remote: {format_bytes(remote_size)})")
                else:
                    correct_files += 1
            
            # Show progress
            print(f"\r  Verified: {total_files} files | ✓ {correct_files} | ✗ {missing_files + size_mismatches}", end='', flush=True)
    
    print()  # New line
    print(f"\nVerification Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Correct: {correct_files}")
    print(f"  Missing: {missing_files}")
    print(f"  Size mismatches: {size_mismatches}")
    
    if missing_files == 0 and size_mismatches == 0:
        print("✓ All files verified successfully!")
    else:
        print("✗ Some files have issues!")
    
    return missing_files + size_mismatches == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast S3 downloader with verification')
    parser.add_argument('--folder', choices=['train', 'extra_las'], required=True, 
                       help='Folder to download (train or extra_las)')
    parser.add_argument('--local-dir', type=str, default=LOCAL_DIR, 
                       help=f'Local directory path (default: {LOCAL_DIR})')
    parser.add_argument('--threads', type=int, default=20, 
                       help='Number of download threads (default: 20)')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify existing files, don\'t download')
    parser.add_argument('--no-verify', action='store_true', 
                       help='Skip verification after download')
    args = parser.parse_args()

    local_directory = args.local_dir
    Path(local_directory).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"S3 Fast Downloader - {args.folder.upper()} folder")
    print(f"Local directory: {local_directory}")
    print(f"Threads: {args.threads}")
    print("=" * 80)
    
    if args.verify_only:
        print("VERIFICATION MODE")
        verify_downloads(BUCKET_NAME, args.folder, local_directory)
    else:
        print("DOWNLOAD MODE")
        download_folder_parallel(BUCKET_NAME, args.folder, local_directory, args.threads)
        
        if not args.no_verify:
            verify_downloads(BUCKET_NAME, args.folder, local_directory)
    
    print("=" * 80)
    print("Complete!")
    print("=" * 80)