import os
import urllib.request
import tarfile

def download_data():
    if not os.path.exists('Data/genres_original'):
        print("Downloading GTZAN dataset from official source (1.2GB)... This will take a while.")
        url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
        os.makedirs('Data', exist_ok=True)
        urllib.request.urlretrieve(url, 'Data/genres.tar.gz')
        print("Download complete. Extracting...")
        with tarfile.open('Data/genres.tar.gz', 'r:gz') as tar:
            tar.extractall(path='Data/genres_original')
        print("Extraction complete.")
        os.remove('Data/genres.tar.gz')
    else:
        print("Dataset already downloaded.")

if __name__ == '__main__':
    download_data()
