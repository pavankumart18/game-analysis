import requests
import os
import sys

def download_file(url, filename):
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        
    print(f"Downloading {url} to {filename}...")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024*1024*10) < 8192: # Print every ~10MB
                            sys.stdout.write(f"\rProgress: {percent:.1f}%")
                            sys.stdout.flush()
                            
        print(f"\nDownload complete: {filename}")
        
    except Exception as e:
        print(f"\nError downloading file: {e}")

if __name__ == "__main__":
    # URL for SAM ViT-H (Huge) - The standard "default" model
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    target = os.path.join("checkpoints", "sam_vit_h_4b8939.pth")
    
    download_file(url, target)
