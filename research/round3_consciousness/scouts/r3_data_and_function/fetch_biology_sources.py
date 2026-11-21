#!/usr/bin/env python3
"""Fetch/verify original public CC BY 4.0 source archives, then safely extract."""
from pathlib import Path
import hashlib
import urllib.request
import zipfile

BASE=Path(__file__).resolve().parent
SOURCES=[
    ("Figure_1.zip","https://zenodo.org/records/19465617/files/Figure%201.zip?download=1"),
    ("Figure_4.zip","https://zenodo.org/records/19465617/files/Figure%204.zip?download=1"),
]

def main():
    # Expected SHA values come from the acquisition manifest, which also records
    # the independent match to the official Zenodo MD5 checksum.
    import json
    manifest=json.loads((BASE/"data_acquisition_manifest.json").read_text())
    checksums={x["file"]:x["sha256"] for x in manifest["files"]}
    dest=BASE/"source_data";dest.mkdir(exist_ok=True)
    for filename,url in SOURCES:
        path=BASE/filename
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        expected=checksums[filename]
        if digest!=expected:
            raise RuntimeError(f"Checksum mismatch for {filename}: {digest} != {expected}")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                candidate=(dest/member.filename).resolve()
                if not candidate.is_relative_to(dest.resolve()):
                    raise RuntimeError(f"Unsafe archive path: {member.filename}")
            archive.extractall(dest)
        print(filename, "verified", digest)

if __name__=="__main__":main()
