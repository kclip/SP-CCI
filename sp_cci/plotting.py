import json, numpy as np
import matplotlib.pyplot as plt
from typing import List

def kde_hist(values: np.ndarray, label: str):
    # Simple histogram w/ density (no seaborn dependency)
    plt.hist(values, bins=30, density=True, alpha=0.4, label=label)

def make_coverage_and_width_plots(json_paths: List[str], outdir: str):
    outdir = outdir.rstrip("/")
    # Coverage
    plt.figure(figsize=(8,4))
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        cov = np.array(data.get("coverage", []), dtype=float)
        if cov.size:
            kde_hist(cov, label=data.get("label", jp))
    plt.axvline(0.85, linestyle="--", label="Target Coverage (example)")
    plt.xlabel("Empirical Coverage")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/Coverage.pdf", bbox_inches="tight")
    plt.close()

    # Width
    plt.figure(figsize=(8,4))
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        wid = np.array(data.get("width", []), dtype=float)
        if wid.size:
            kde_hist(wid, label=data.get("label", jp))
    plt.xlabel("Average Interval Width")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/prediction_width.pdf", bbox_inches="tight")
    plt.close()
