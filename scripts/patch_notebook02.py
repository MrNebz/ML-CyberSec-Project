"""One-time patch: append deployment-artifact save cells to Notebook 02.

Adds two cells at the end of 02_subset_splitting_and_preprocessing.ipynb:
  1. Markdown explainer (Step 15)
  2. Code that saves iqr_bounds.json + feature_names.json to data/artifacts/

Run from project root:
    python scripts/patch_notebook02.py

Idempotent: skips if the cells are already present.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_subset_splitting_and_preprocessing.ipynb"

MARKDOWN_MARKER = "# Step 15 — Save Deployment Inference Artifacts"
CODE_MARKER = "iqr_bounds.json"

MARKDOWN_SOURCE = [
    "# Step 15 — Save Deployment Inference Artifacts\n",
    "\n",
    "The model-serving service in `src/serve/` needs to reproduce the *exact*\n",
    "preprocessing applied during training when it receives a new feature vector.\n",
    "Two small artifacts are missing from Step 14 that we add here:\n",
    "\n",
    "| Artifact | Why it is needed at inference time |\n",
    "|---|---|\n",
    "| `iqr_bounds.json` | The per-feature `lower`/`upper` bounds used to clip outliers on the `out` and `out_scaled` variants. Without these, the service cannot apply the same clipping step. |\n",
    "| `feature_names.json` | The ordered list of the 39 feature columns. The service uses this to validate JSON requests and put columns in the exact order the model expects. |\n",
    "\n",
    "Both files are additive — nothing already saved in Step 14 changes."
]

CODE_SOURCE = [
    "# Save IQR bounds (for out / out_scaled inference) and feature column order.\n",
    "iqr_bounds = {\n",
    "    \"clip_candidates\": list(clip_candidates),\n",
    "    \"lower\": {col: float(lower_bounds[col]) for col in clip_candidates},\n",
    "    \"upper\": {col: float(upper_bounds[col]) for col in clip_candidates},\n",
    "}\n",
    "\n",
    "with open(ARTIFACTS_DIR / \"iqr_bounds.json\", \"w\") as f:\n",
    "    json.dump(iqr_bounds, f, indent=2)\n",
    "print(\"Saved iqr_bounds.json\")\n",
    "\n",
    "feature_names = list(X_train_raw.columns)\n",
    "with open(ARTIFACTS_DIR / \"feature_names.json\", \"w\") as f:\n",
    "    json.dump(feature_names, f, indent=2)\n",
    "print(f\"Saved feature_names.json ({len(feature_names)} features)\")"
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    def source_contains(cell: dict, marker: str) -> bool:
        src = cell.get("source", [])
        joined = "".join(src) if isinstance(src, list) else str(src)
        return marker in joined

    already_patched = any(
        source_contains(c, MARKDOWN_MARKER) or source_contains(c, CODE_MARKER)
        for c in nb["cells"]
    )
    if already_patched:
        print("Notebook already patched — no changes made.")
        return

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": MARKDOWN_SOURCE})
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_SOURCE,
    })

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Appended 2 cells to {NB_PATH.name}.")


if __name__ == "__main__":
    main()
