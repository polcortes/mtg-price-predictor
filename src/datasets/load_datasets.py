import os
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import zipfile
from pathlib import Path


BASE = "https://mtgjson.com/api/v5/"
DATASETS = ("AllPrintings.json", "AllPrices.json")
OUT_DIR = Path(__file__).resolve().parent
CHUNK_SIZE = 1 << 20  # 1 MiB

def session_with_retries():
    """Create and return a `requests.Session` configured with automatic retries.

    Builds a session with urllib3's `Retry` strategy for resilient HTTP
    requests. Retries are applied to GET requests only, covering transient
    server errors (5xx) and rate-limiting (429). The session also sets a
    custom User-Agent and enables gzip/deflate compression.

    Returns:
        requests.Session: A session instance with retry adapters mounted for
        both ``http://`` and ``https://`` URLs.

    Raises:
        requests.exceptions.RetryError: If all retry attempts are exhausted
        for a given request.
    """
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

def download_file(s, url, dest_path: Path):
    """Stream-download a file from *url* and save it to *dest_path*.

    Downloads the remote resource using a streaming GET request and writes
    it to a temporary ``.part`` file first, then atomically moves it to the
    final destination. Progress is printed to stdout at every 10% interval
    when the server provides a ``Content-Length`` header.

    Args:
        s (requests.Session): A requests session to use for the download.
        url (str): The URL of the file to download.
        dest_path (str): The local filesystem path where the file should be
            saved.
    """

    tmp_path = Path(dest_path.__str__() + '.part')
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
    """Download, extract, and clean up all configured datasets.

    Iterates over the module-level ``DATASETS`` list, downloads each dataset
    as a ZIP archive from ``BASE``, extracts the single JSON file named
    after the dataset into ``OUT_DIR``, then deletes the ZIP to save disk
    space.

    Side Effects:
        - Writes ``{OUT_DIR}/{ds}.zip`` for each dataset during download.
        - Writes ``{OUT_DIR}/{ds}`` (the extracted file) for each dataset.
        - Deletes each ZIP archive after successful extraction.
        - Prints progress messages to ``stdout``.

    Raises:
        requests.exceptions.RequestException: Propagated from
            :func:`download_file` if a download fails.
        zipfile.BadZipFile: If a downloaded archive is corrupted or not a
            valid ZIP file.
        KeyError: If the ZIP archive does not contain a member exactly
            named ``ds``.
        OSError: If file system operations (write, extract, unlink) fail.
    """
    s = session_with_retries()
    for ds in DATASETS:
        zip_url = f"{BASE}{ds}.zip"
        zip_path = OUT_DIR / f"{ds}.zip"
        json_path = OUT_DIR / ds
        print("Downloading", ds, "as ZIP")
        download_file(s, zip_url, zip_path)
        print("Extracting", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            # Assumes the ZIP contains exactly {ds}
            z.extract(ds, OUT_DIR)

        # remove ZIP to save space
        zip_path.unlink()
        print("Saved", json_path)

if __name__ == "__main__":
    load_datasets()
