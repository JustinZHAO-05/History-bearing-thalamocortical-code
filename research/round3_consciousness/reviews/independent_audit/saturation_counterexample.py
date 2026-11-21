"""A constructive identifiability counterexample, with no fitted parameters.

One stored scalar h and a transient sensory drive s feed two clipped rectified
rate units. Their fixed linear difference is the output. The source write flips
h before either response, and the fifth sensory drive weakly updates h before
the later ambiguous response. There is no instruction/operation gate.
"""
from pathlib import Path
import json
import numpy as np


def output(h, s):
    v = h + s
    positive_rate = np.clip(v, 0, 1)
    negative_rate = np.clip(-v, 0, 1)
    return positive_rate - negative_rate


def main():
    rows = []
    for historical_target in (-1, 1):
        for current_target in (-1, 1):
            for source_multiplier in (1, -1):
                h = .85 * historical_target * source_multiplier
                s = 2 * current_target
                current_output = float(output(h, s))
                h += .025 * s
                history_output = float(output(h, 0))
                rows.append(dict(historical_target=historical_target, current_target=current_target,
                                 source_multiplier=source_multiplier, current_output=current_output,
                                 history_output=history_output,
                                 current_committed_correct=current_output*current_target >= .75,
                                 history_committed_with_source=history_output*historical_target*source_multiplier >= .75))
    assert all(r["current_committed_correct"] and r["history_committed_with_source"] for r in rows)
    result = dict(
        interpretation="The exact strong-current-correct plus later-history-reversed signature can occur with one stored historical scalar and saturation, a fixed linear readout of clipped rectified rates, and no independently varying operation gate.",
        status="Logical counterexample, not a fitted model of any checkpoint",
        rate_units="r_plus=clip(h+s,0,1); r_minus=clip(-(h+s),0,1); y=r_plus-r_minus",
        memory_update="h <- h + 0.025*s after the current decision; later ambiguous s=0",
        rows=rows,
    )
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
