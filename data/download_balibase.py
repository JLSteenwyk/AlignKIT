"""Download and extract BAliBASE 3 benchmark dataset."""

import tarfile
import urllib.request
import shutil
from pathlib import Path

# Primary and fallback download URLs for BAliBASE 3
URLS = [
    "https://web.archive.org/web/2024/https://lbgi.fr/balibase/BalibaseDownload/BAliBASE_R1-5.tar.gz",
    "https://lbgi.fr/balibase/BalibaseDownload/BAliBASE_R1-5.tar.gz",
    "https://lbgi.fr/balibase/BalibaseDownload/bb3_release.tar.gz",
]

EXPECTED_REF_SETS = ["RV11", "RV12", "RV20", "RV30", "RV40", "RV50"]


def download_balibase(dest_dir: Path = None):
    """Download and extract BAliBASE 3 to dest_dir."""
    if dest_dir is None:
        dest_dir = Path(__file__).parent / "bb3_release"

    # Check if already downloaded
    if dest_dir.exists():
        existing = [d.name for d in dest_dir.iterdir() if d.is_dir()]
        if any(rv in existing for rv in EXPECTED_REF_SETS):
            print(f"BAliBASE already present at {dest_dir}")
            _verify(dest_dir)
            return dest_dir

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir.parent / "balibase3.tar.gz"

    # Try each URL
    for url in URLS:
        print(f"Trying to download from {url} ...")
        try:
            urllib.request.urlretrieve(url, tar_path)
            print(f"Downloaded to {tar_path}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        raise RuntimeError(
            "Could not download BAliBASE 3 from any known URL.\n"
            "Please download manually and extract to: " + str(dest_dir)
        )

    # Extract
    print("Extracting archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir.parent)

    # The archive may extract to a subdirectory — find it
    if not dest_dir.exists():
        # Look for extracted directory containing reference sets
        for child in dest_dir.parent.iterdir():
            if child.is_dir() and child != dest_dir:
                child_contents = [d.name for d in child.iterdir() if d.is_dir()]
                if any(rv in child_contents for rv in EXPECTED_REF_SETS):
                    shutil.move(str(child), str(dest_dir))
                    break
                # Check one level deeper
                for subchild in child.iterdir():
                    if subchild.is_dir():
                        sub_contents = [d.name for d in subchild.iterdir() if d.is_dir()]
                        if any(rv in sub_contents for rv in EXPECTED_REF_SETS):
                            shutil.move(str(subchild), str(dest_dir))
                            break

    # Clean up tarball
    tar_path.unlink(missing_ok=True)

    _verify(dest_dir)
    return dest_dir


def _verify(dest_dir: Path):
    """Verify that expected reference set directories exist."""
    found = []
    missing = []
    for rv in EXPECTED_REF_SETS:
        if (dest_dir / rv).is_dir():
            found.append(rv)
        else:
            missing.append(rv)

    print(f"Found reference sets: {found}")
    if missing:
        print(f"WARNING: Missing reference sets: {missing}")
    else:
        print("All reference sets present.")


if __name__ == "__main__":
    download_balibase()
