import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.reader_interference import measure_readout_interference
from src.models.linear_feature_encoder import LinearFeatureEncoder
from src.utils import seed_everything


CONFIG = {
    "embedding_dim": 64,
    "output_dim": 16,
    "step_size": 0.1,
    "alignments": [0.0, 0.5, 1.0],
    "atol": 1e-5,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check interference through frozen linear readers.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    seed_everything(args.seeds[0])
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    d, k = CONFIG["embedding_dim"], CONFIG["output_dim"]
    identity = torch.eye(k, dtype=torch.float32, device=args.device)
    cases = []

    for seed in args.seeds:
        seed_everything(seed)
        # Dense rotation avoids privileging coordinate-aligned subspaces.
        basis, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device=args.device))
        u, v, w = basis.T[:3 * k].split(k)
        encoder = LinearFeatureEncoder({name: d for name in ("user_1", "user_2", "item")}, k)
        encoder = encoder.to(args.device).requires_grad_(False)
        embedding = torch.empty(d, dtype=torch.float32, device=args.device).uniform_(-0.05, 0.05)

        for alignment in CONFIG["alignments"]:
            remainder = math.sqrt(1 - alignment**2)
            weights = {
                "user_1": u,
                "user_2": alignment * u + remainder * v,
                "item": alignment * u + remainder * w,
            }
            readers = {}
            for name, weight in weights.items():
                encoder.readers[name].weight.copy_(weight)
                singular_values = torch.linalg.svdvals(weight)
                readers[name] = {
                    "rank": torch.linalg.matrix_rank(weight).item(),
                    "singular_min": singular_values.min().item(),
                    "singular_max": singular_values.max().item(),
                    "row_gram_max_error": (weight @ weight.T - identity).abs().max().item(),
                }

            checks = {}
            for label, source in (("self", "user_1"), ("inter_feature", "user_2"), ("cross_tower", "item")):
                check = measure_readout_interference(
                    encoder.readers["user_1"], encoder.readers[source], embedding, CONFIG["step_size"],
                )
                expected_gain = 1.0 if source == "user_1" else alignment
                check["expected_gain"] = expected_gain
                check["passed"] = (
                    check["formula_error_per_update"] < CONFIG["atol"]
                    and abs(check["measured_gain"] - expected_gain) < CONFIG["atol"]
                    and abs(check["overlap_per_rank"] - expected_gain**2) < CONFIG["atol"]
                    and abs(check["update_l2"] - CONFIG["step_size"] * math.sqrt(k)) < CONFIG["atol"]
                )
                checks[label] = check
            passed = all(check["passed"] for check in checks.values()) and all(
                reader["rank"] == k and reader["row_gram_max_error"] < CONFIG["atol"]
                for reader in readers.values()
            )
            cases.append({"seed": seed, "alignment": alignment, "readers": readers, "checks": checks, "passed": passed})
            print(json.dumps({
                "seed": seed, "alignment": alignment, "passed": passed,
                "measured_gain": {name: check["measured_gain"] for name, check in checks.items()},
            }), flush=True)

    torch.cuda.synchronize(args.device)
    report = {
        "config": CONFIG,
        "seeds": args.seeds,
        "dtype": "float32",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(args.device) / 2**20,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir or root / "experiment_logs/theory/reader_interference" / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Artifacts: {output.resolve()}", flush=True)
    if not report["passed"]:
        raise RuntimeError(f"Frozen-reader interference check failed: {output / 'result.json'}")
    print(f"CHECK COMPLETE: {report['elapsed_s']:.2f}s, {report['peak_allocated_mib']:.2f} MiB allocated", flush=True)


if __name__ == "__main__":
    main()
