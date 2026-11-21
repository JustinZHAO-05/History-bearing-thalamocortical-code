"""Fetch the published behavioral inputs and original model checkpoints."""
import argparse
from hashlib import file_digest
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent / 'research/round3_consciousness'
SCOUT = ROOT / 'scouts/r3_data_and_function'
MODEL_URL = 'https://zenodo.org/records/19465617/files/rnn_model.zip?download=1'


def prepare_models():
    spec = importlib.util.spec_from_file_location('portable_reproduce', SCOUT / 'portable_reproduce.py')
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    archive = ROOT / 'data/hosford_model_source.zip'
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.part')
        request = Request(MODEL_URL, headers={'User-Agent': 'history-bearing-thalamocortical-code/1.0'})
        try:
            with urlopen(request, timeout=90) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            with temporary.open('rb') as input_file:
                if file_digest(input_file, 'sha256').hexdigest() != portable.ARCHIVE_SHA256:
                    raise ValueError('The downloaded model archive checksum did not match the expected input.')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    subprocess.run([sys.executable, str(SCOUT / 'portable_reproduce.py'),
                    'position_mixture', '--prepare-only'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', choices=('models', 'behavior', 'all'), default='all', nargs='?')
    options = parser.parse_args()
    if options.inputs in ('models', 'all'):
        prepare_models()
    if options.inputs in ('behavior', 'all'):
        subprocess.run([sys.executable, str(SCOUT / 'portable_reproduce.py'),
                        'biology', '--prepare-only'], check=True)


if __name__ == '__main__':
    main()
