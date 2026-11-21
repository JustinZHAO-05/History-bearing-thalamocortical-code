#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent/'research/final_round/code/reproduce_final.py'),run_name='__main__')
