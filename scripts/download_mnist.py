"""Download and verify the original MNIST IDX files used by the notebooks."""

import argparse
import gzip
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte": (
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    ),
    "train-labels-idx1-ubyte": (
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    ),
    "t10k-images-idx3-ubyte": (
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    ),
    "t10k-labels-idx1-ubyte": (
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(destination, name, compressed_hash, raw_hash):
    raw_path = destination / name
    if raw_path.exists() and sha256(raw_path) == raw_hash:
        print(f"verified {raw_path}")
        return

    compressed_path = destination / f".{name}.gz.part"
    output_path = destination / f".{name}.part"
    try:
        with urllib.request.urlopen(BASE_URL + name + ".gz", timeout=60) as response:
            with compressed_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        if sha256(compressed_path) != compressed_hash:
            raise ValueError(f"download checksum mismatch: {name}")
        with gzip.open(compressed_path, "rb") as source:
            with output_path.open("wb") as output:
                shutil.copyfileobj(source, output)
        if sha256(output_path) != raw_hash:
            raise ValueError(f"dataset checksum mismatch: {name}")
        os.replace(output_path, raw_path)
        print(f"downloaded {raw_path}")
    finally:
        compressed_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "mnist",
        help="destination directory (default: data/mnist in this repository)",
    )
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, hashes in FILES.items():
        download_one(args.dest, name, *hashes)


if __name__ == "__main__":
    main()
