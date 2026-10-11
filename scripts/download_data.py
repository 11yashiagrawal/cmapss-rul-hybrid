"""Download NASA C-MAPSS raw files using only the Python standard library."""

from io import BytesIO
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "CMAPSSData"
FILES = tuple(
    f"{kind}_FD{subset:03}.txt"
    for subset in range(1, 5)
    for kind in ("train", "test", "RUL")
)


def download_data() -> None:
    """Download and extract all four subsets, skipping complete local data."""
    if all((DATA_DIR / name).is_file() for name in FILES):
        print(f"C-MAPSS files already exist in {DATA_DIR}; skipping download.")
        return

    print(f"Downloading C-MAPSS from {URL}")
    with TemporaryFile() as downloaded:
        with urlopen(URL, timeout=60) as response:
            copyfileobj(response, downloaded)
        downloaded.seek(0)
        with ZipFile(downloaded) as bundle:
            inner = next(
                name for name in bundle.namelist()
                if name.endswith("CMAPSSData.zip")
            )
            raw_archive = bundle.read(inner)
        with ZipFile(BytesIO(raw_archive)) as archive:
            members = {Path(name).name: name for name in archive.namelist()}
            missing = set(FILES) - members.keys()
            if missing:
                raise ValueError(
                    f"C-MAPSS archive is missing: {', '.join(sorted(missing))}"
                )
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Write only the expected filenames, never archive-supplied paths.
            for name in FILES:
                (DATA_DIR / name).write_bytes(archive.read(members[name]))
    print(f"Extracted C-MAPSS files to {DATA_DIR}")


if __name__ == "__main__":
    download_data()
