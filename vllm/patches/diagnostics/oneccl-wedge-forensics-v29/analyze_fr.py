"""llm-scaler v29: partition fr-log AR markers by spec-decode region.

Reads /tmp/fr_<pid>.log (v28dbg recorder + v29 propose markers) and
reports, per rank: total propose calls, ARs logged INSIDE propose (drafter
region) vs OUTSIDE (target verify + sampler + misc), plus the per-step
averages implied by the chunk count. Decides where the graphs-x-spec
eager-collective surface actually lives.
"""
import glob
import sys

for path in sorted(glob.glob("/tmp/fr_*.log")):
    in_propose = False
    ar_in = ar_out = 0
    propose_calls = 0
    with open(path) as fh:
        for line in fh:
            if line.endswith("propose begin\n"):
                in_propose = True
                propose_calls += 1
            elif line.endswith("propose end\n"):
                in_propose = False
            elif " AR begin" in line:
                if in_propose:
                    ar_in += 1
                else:
                    ar_out += 1
    print(f"{path}: propose={propose_calls} AR_in_propose={ar_in} "
          f"AR_outside={ar_out} "
          f"AR/propose_call={ar_in / max(propose_calls, 1):.2f}")
