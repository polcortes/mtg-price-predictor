import os
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import zipfile

BASE = "https://mtgjson.com/api/v5/"
DATASETS = ("AllPrintings.json", "AllPricesToday.json")
OUT_DIR = "./datasets"
CHUNK_SIZE = 1 << 20  # 1 MiB

def session_with_retries():
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    s.headers.update({
        "User-Agent": "mtg-price-predictor/1.0 (+https://github.com/polcortes/mtg-price-predictor)",
        "Accept-Encoding": "gzip, deflate",
    })
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def download_file(s, url, dest_path):
    tmp_path = dest_path + ".part"
    with s.get(url, stream=True, timeout=(10, 600)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        next_mark = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    if total:
                        downloaded += len(chunk)
                        pct = int(downloaded * 100 / total)
                        if pct >= next_mark:
                            print(f"Progress {pct}% ({downloaded:,}/{total:,} bytes)")
                            next_mark += 10
    os.replace(tmp_path, dest_path)

def load_datasets():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = session_with_retries()
    for ds in DATASETS:
        zip_url = f"{BASE}{ds}.zip"
        zip_path = os.path.join(OUT_DIR, f"{ds}.zip")
        json_path = os.path.join(OUT_DIR, ds)
        print("Downloading", ds, "as ZIP")
        download_file(s, zip_url, zip_path)
        print("Extracting", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            # Assumes the ZIP contains exactly {ds}
            z.extract(ds, OUT_DIR)

        # Optional: remove ZIP to save space
        os.remove(zip_path)
        print("Saved", json_path)

if __name__ == "__main__":
    load_datasets()
