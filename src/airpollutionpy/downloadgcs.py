import os
from google.cloud import storage

def download_bucket_files(bucket_name, output_folder, prefix=None):
    """
    Downloads files from a Google Cloud Storage bucket to a local output folder,
    skipping files that already exist locally.
    
    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket (without gs:// prefix)
        output_folder (str): Path to local output folder
        prefix (str, optional): Filter results to objects with this prefix (folder path)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    try:
        # Create a client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs in the bucket with the specified prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            print(f"No files found in bucket '{bucket_name}' with prefix '{prefix}'")
            return
        
        total_files = len(blobs)
        downloaded = 0
        skipped = 0
        
        print(f"Found {total_files} files in bucket")
        print(output_folder)
        # Process each blob in the bucket
        for blob in blobs:
            # Skip "folder" objects
            if blob.name.endswith('/'):
                continue

            print(output_folder)
            local_file_path = os.path.join(output_folder, os.path.basename(blob.name))
            
            # Check if file already exists locally
            if os.path.exists(local_file_path):
                # Get local file size
                local_size = os.path.getsize(local_file_path)
                
                # Get blob size
                if local_size == blob.size:
                    print(f"Skipping {blob.name} (already exists)")
                    skipped += 1
                    continue
            
            # File doesn't exist locally or has different size, download it
            print(f"Downloading {blob.name}...")
            blob.download_to_filename(local_file_path)
            downloaded += 1
            
        print(f"Download complete: {downloaded} files downloaded, {skipped} files skipped")
    
    except Exception as e:
        print(f"Error: {e}")