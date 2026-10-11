"""Regression tests for the standalone C-MAPSS downloader."""

from io import BytesIO
from pathlib import Path
import runpy
from zipfile import ZipFile

import pytest

EXPECTED_FILES = (
    "train_FD001.txt test_FD001.txt RUL_FD001.txt "
    "train_FD002.txt test_FD002.txt RUL_FD002.txt "
    "train_FD003.txt test_FD003.txt RUL_FD003.txt "
    "train_FD004.txt test_FD004.txt RUL_FD004.txt"
).split()


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    download = runpy.run_path(str(script))["download_data"]
    monkeypatch.setitem(download.__globals__, "DATA_DIR", tmp_path / "CMAPSSData")
    return download


def nasa_archive(names):
    raw = BytesIO()
    with ZipFile(raw, "w") as archive:
        for name in names:
            archive.writestr(f"dataset/{name}", f"contents of {name}")
        archive.writestr("../../unexpected.txt", "must not be extracted")
    bundle = BytesIO()
    with ZipFile(bundle, "w") as archive:
        archive.writestr("NASA/CMAPSSData.zip", raw.getvalue())
    bundle.seek(0)
    return bundle


@pytest.mark.parametrize("incomplete", [False, True])
def test_download_extracts_only_expected_files_and_skips_complete_data(
    downloader, tmp_path, monkeypatch, incomplete
):
    destination = tmp_path / "CMAPSSData"
    if incomplete:
        destination.mkdir()
        (destination / "train_FD001.txt").write_text("old contents")
    response = nasa_archive(EXPECTED_FILES)
    monkeypatch.setitem(downloader.__globals__, "urlopen", lambda *args, **kwargs: response)

    downloader()

    assert {path.name for path in destination.iterdir()} == set(EXPECTED_FILES)
    assert not (tmp_path / "unexpected.txt").exists()
    for name in EXPECTED_FILES:
        assert (destination / name).read_text() == f"contents of {name}"

    def unexpected_download(*args, **kwargs):
        pytest.fail("Complete data must not trigger another download")

    monkeypatch.setitem(downloader.__globals__, "urlopen", unexpected_download)
    downloader()


def test_missing_archive_file_is_rejected_before_writing(downloader, tmp_path, monkeypatch):
    response = nasa_archive(EXPECTED_FILES[:-1])
    monkeypatch.setitem(downloader.__globals__, "urlopen", lambda *args, **kwargs: response)

    with pytest.raises(ValueError, match="RUL_FD004.txt"):
        downloader()

    assert not (tmp_path / "CMAPSSData").exists()
