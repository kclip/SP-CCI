import json, numpy as np

def save_results(path, label, coverage_list, width_list, config=None):
    obj = {
        "label": label,
        "coverage": list(map(float, coverage_list)),
        "width": list(map(float, width_list)),
        "config": config or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}")
