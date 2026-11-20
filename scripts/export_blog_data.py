#!/usr/bin/env python3
"""Export verified experimental data from archived CSVs and JSON results for the interactive blog."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments/subspace_study/results"
LOGS_IN1K = REPO_ROOT / "experiments/subspace_study/logs/imagenet1k"
OUT_PATH = REPO_ROOT / "docs/assets/blog_data.json"


def _read_curve(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_raw = row.get("epoch", "").strip()
            if not ep_raw:
                continue
            ep = int(float(ep_raw))
            acc1_s = row.get("val/acc1", "").strip()
            acc5_s = row.get("val/acc5", "").strip()
            loss_s = row.get("val/loss", "").strip()

            def _parse(v: str) -> float | None:
                if not v or v.lower() in ("nan", "inf", "-inf"):
                    return None
                val = float(v)
                return None if (math.isnan(val) or math.isinf(val)) else val

            acc1 = _parse(acc1_s)
            acc5 = _parse(acc5_s)
            loss = _parse(loss_s)
            rows.append(
                {
                    "e": ep,
                    "a1": round(acc1 * 100.0, 2) if acc1 is not None else None,
                    "a5": round(acc5 * 100.0, 2) if acc5 is not None else None,
                    "loss": round(loss, 4) if loss is not None else None,
                    "nan": (loss is None and bool(loss_s)) or (loss is None and acc1 is not None),
                }
            )
    return rows


def main() -> None:
    wave_c = json.loads((RESULTS_DIR / "summary_in1k_waveC.json").read_text())
    wave_d = json.loads((RESULTS_DIR / "summary_in1k_waveD.json").read_text())
    wave_f = json.loads((RESULTS_DIR / "summary_in1k_waveF.json").read_text())
    e2e_bench = json.loads((RESULTS_DIR / "e2e_triton_benchmark_h800.json").read_text())
    kernel_bench = json.loads((RESULTS_DIR / "kernel_benchmark_h800.json").read_text())

    # 1. 100-epoch ImageNet-1k Atto comparison
    atto_100ep: list[dict[str, Any]] = []
    name_map = {
        "in1k_v2_atto_k4_100ep": ("Subspace-CAN-v2-Atto (Full)", "full", 71.25),
        "in1k_v2_atto_inner_wide_100ep": ("Subspace-CAN-v2-Atto (Inner-wide)", "inner", 71.33),
        "in1k_v2_rung2_deadzone_100ep": ("Subspace-CAN-v2-Rung2", "rung2", 72.92),
        "in1k_convnext_atto_100ep": ("ConvNeXt-Atto", "convnext", 71.20),
    }
    for item in wave_c:
        rid = item["run"]
        label, tag, ema_top1 = name_map[rid]
        ev = item["eval"]
        corr = ev["corruptions"]
        diag = ev.get("diagnostics", {})
        atto_100ep.append(
            {
                "id": rid,
                "label": label,
                "tag": tag,
                "params_m": ev["params_m"],
                "gmacs": item["gmacs"],
                "clean_top1": ev["clean_top1"],
                "ema_top1": ema_top1,
                "clean_top5": ev["clean_top5"],
                "contrast_012": corr["contrast_0.12"],
                "gaussian_noise": corr["gaussian_noise_0.08"],
                "lowpass_blur": corr["lowpass_blur"],
                "fft_phase": corr["fft_phase_only"],
                "effective_rank": round(diag.get("spectrum/effective_rank", 0.0), 3) if diag else None,
            }
        )

    # 2. 90-epoch Pico trajectories & summary stats
    pico_curves = {
        "convnext_pico": {
            "label": "ConvNeXt-Pico (9.05M, LN2d)",
            "color": "#4b5563",
            "params_m": wave_d["in1k_convnext_pico_90ep"]["params_m"],
            "gmacs": 1.37,
            "clean_top1": wave_d["in1k_convnext_pico_90ep"]["clean_top1"],
            "clean_top5": wave_d["in1k_convnext_pico_90ep"]["clean_top5"],
            "nan_epochs": 0,
            "max_wedge": None,
            "lr": "4.0e-3",
            "cache": "square256",
            "points": _read_curve(LOGS_IN1K / "in1k_convnext_pico_90ep_metrics.csv"),
        },
        "pico_bn": {
            "label": "Subspace-CAN-v2-Pico (BN2d, 8.50M)",
            "color": "#c94136",
            "params_m": wave_d["in1k_v2_pico_k4_90ep"]["params_m"],
            "gmacs": 1.32,
            "clean_top1": wave_d["in1k_v2_pico_k4_90ep"]["clean_top1"],
            "clean_top5": wave_d["in1k_v2_pico_k4_90ep"]["clean_top5"],
            "nan_epochs": 34,
            "max_wedge": 3466.0,
            "lr": "2.0e-3",
            "cache": "square256",
            "points": _read_curve(LOGS_IN1K / "in1k_v2_pico_k4_90ep_metrics.csv"),
        },
        "pico_rmsnorm": {
            "label": "Subspace-CAN-v2-Pico (RMSNorm2d, 8.49M)",
            "color": "#0f766e",
            "params_m": wave_f["in1k_v2_pico_rmsnorm2d_fast_lr4e3_90ep"]["eval"]["params_m"],
            "gmacs": 1.32,
            "clean_top1": wave_f["in1k_v2_pico_rmsnorm2d_fast_lr4e3_90ep"]["eval"]["clean_top1"],
            "clean_top5": wave_f["in1k_v2_pico_rmsnorm2d_fast_lr4e3_90ep"]["eval"]["clean_top5"],
            "nan_epochs": 0,
            "max_wedge": 64.4,
            "lr": "4.0e-3",
            "cache": "ar320",
            "points": _read_curve(LOGS_IN1K / "in1k_v2_pico_rmsnorm2d_fast_lr4e3_90ep_metrics.csv"),
        },
        "pico_rmsnorm_sphere": {
            "label": "Subspace-CAN-v2-Pico (RMSNorm2d + Sphere)",
            "color": "#b87314",
            "params_m": wave_f["in1k_v2_pico_rmsnorm2d_sphere_fast_lr4e3_90ep"]["eval"]["params_m"],
            "gmacs": 1.32,
            "clean_top1": wave_f["in1k_v2_pico_rmsnorm2d_sphere_fast_lr4e3_90ep"]["eval"]["clean_top1"],
            "clean_top5": wave_f["in1k_v2_pico_rmsnorm2d_sphere_fast_lr4e3_90ep"]["eval"]["clean_top5"],
            "nan_epochs": 0,
            "max_wedge": 24.3,
            "lr": "4.0e-3",
            "cache": "ar320",
            "points": _read_curve(LOGS_IN1K / "in1k_v2_pico_rmsnorm2d_sphere_fast_lr4e3_90ep_metrics.csv"),
        },
    }

    # 3. Single-op microbenchmark extract: paper (64, 128, 64, 64), bf16, NHWC, k in {2,3,4}
    micro_bf16_nhwc: list[dict[str, Any]] = []
    for entry in kernel_bench.get("benchmarks", []):
        if (
            entry.get("label") == "paper_64x128x64x64"
            and entry.get("dtype") == "bf16"
            and entry.get("memory_format") == "NHWC"
        ):
            res_map = entry.get("results", {})
            for impl in ("pytorch_eager", "torch_compile_default", "triton_recompute"):
                if impl not in res_map:
                    continue
                r = res_map[impl]
                micro_bf16_nhwc.append(
                    {
                        "k": entry["k"],
                        "impl": impl,
                        "fwd_ms": round(r["fwd_ms"], 3),
                        "bwd_ms": round(r["bwd_ms"], 3),
                        "total_ms": round(r["total_ms"], 3),
                        "peak_mib": round(r["peak_extra_allocated_mib"], 1),
                    }
                )

    # 4. Placeholder for matched 2x2 CIFAR-100 (will be overwritten once Wave 5 finishes!)
    wave5_path = RESULTS_DIR / "summary_cifar100_wave5_matched_2x2.json"
    if wave5_path.exists():
        matched_2x2 = json.loads(wave5_path.read_text())
    else:
        matched_2x2 = {"status": "pending"}

    payload = {
        "atto_100ep": atto_100ep,
        "pico_90ep": pico_curves,
        "wave_e_matched_stability": [
            {"name": "Pico BN2d (Wave D, lr=2e-3, sq256)", "epochs_run": 90, "nan_epochs": 34, "max_wedge": 3466.0},
            {"name": "Nano BN2d (Wave D, lr=2e-3, sq256)", "epochs_run": 90, "nan_epochs": 38, "max_wedge": 5156.0},
            {"name": "Pico BN2d + One-Sided Sphere (Wave E, lr=2e-3, sq256)", "epochs_run": 64, "nan_epochs": 0, "max_wedge": 55.7},
            {"name": "Pico RMSNorm2d (Wave E, lr=2e-3, sq256)", "epochs_run": 42, "nan_epochs": 0, "max_wedge": 58.2},
            {"name": "Pico RMSNorm2d Fast (Wave F, lr=4e-3, ar320)", "epochs_run": 90, "nan_epochs": 0, "max_wedge": 64.4},
            {"name": "Pico RMSNorm2d + Sphere (Wave F, lr=4e-3, ar320)", "epochs_run": 90, "nan_epochs": 0, "max_wedge": 24.3},
        ],
        "e2e_triton": e2e_bench["results"],
        "micro_triton_bf16_nhwc": micro_bf16_nhwc,
        "matched_2x2_cifar100": matched_2x2,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    OUT_PATH.write_text(text, encoding="utf-8")
    js_path = OUT_PATH.with_suffix(".js")
    js_path.write_text(f"window.__BLOG_DATA__ = {text};\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes) and {js_path}")


if __name__ == "__main__":
    main()
