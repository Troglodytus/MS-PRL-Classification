#!/usr/bin/env python3
import os
import re
from pathlib import Path
import pandas as pd

# -------- helpers --------
_TE_BASENAME_RE = re.compile(r"(?:[_-]TE(\d+))", re.IGNORECASE)

def _te_from_row(row):
    """
    Try multiple sources to infer echo/TE ordering:
    1) 'filename' pattern ..._TE04_...
    2) numeric column 'TE_ms'
    Falls back to None if not available.
    """
    # 1) filename pattern
    fn = str(row.get("filename", "")) or ""
    m = _TE_BASENAME_RE.search(fn)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # 2) numeric TE_ms
    te_ms = row.get("TE_ms", None)
    try:
        return float(te_ms) if pd.notna(te_ms) else None
    except Exception:
        return None

def _normalize_modality_name(mod):
    """
    Map your catalogue's modality labels to consistent channel keys.
    Adjust this map to your data conventions.
    """
    if not isinstance(mod, str):
        return None
    m = mod.strip().lower()
    # common aliases
    if m in {"t2", "t2w", "t2-weighted"}:          return "T2"
    if m in {"flair", "t2_flair", "t2f"}:          return "FLAIR"
    if m in {"swi", "swi_mag", "swi-composed"}:    return "SWI"
    if m in {"flair-swi", "flairswi"}:             return "FLAIR-SWI"
    if m in {"qsm_paramagnetic", "paramag", "pos"}:return "Paramagnetic"
    if m in {"qsm_diamagnetic", "diamag", "neg"}:  return "Diamagnetic"
    if m in {"amag", "magnitude", "gre_mag"}:      return "AMag"
    if m in {"aphase", "phase", "gre_phase"}:      return "APhase"
    # keep as-is (capitalized) if unknown
    return mod.strip()

def _select_by_te(rows_df, policy="max", closest_ms=None):
    """
    Given several rows for the *same* (patient, modality), select one by TE policy.
    policy: 'max' | 'min' | 'closest'
    """
    df = rows_df.copy()
    df["__te"] = df.apply(_te_from_row, axis=1)

    if policy == "max":
        df = df.sort_values(["__te"], ascending=True)  # NaN go last
        chosen = df.iloc[-1]
    elif policy == "min":
        df = df.sort_values(["__te"], ascending=True)
        # first non-nan if present
        non_nan = df[df["__te"].notna()]
        chosen = (non_nan.iloc[0] if len(non_nan) else df.iloc[0])
    elif policy == "closest":
        if closest_ms is None:
            raise ValueError("closest_ms must be provided when policy='closest'.")
        df["__dist"] = (df["__te"] - closest_ms).abs()
        non_nan = df[df["__te"].notna()]
        chosen = (non_nan.sort_values(["__dist"]).iloc[0] if len(non_nan) else df.iloc[0])
    else:
        raise ValueError(f"Unknown TE selection policy: {policy}")
    return chosen

# -------- main API --------
def build_patient_modalities_from_excel(
    excel_name: str = "MS_Data_Catalogue.xlsx",
    image_type_required: str = "crosscorrelated",
    allowed_modalities: list | None = None,
    te_policy: str = "max",         # 'max' (default), 'min', or 'closest'
    te_closest_ms: float | None = None,
    drop_missing_files: bool = True,
):
    """
    Returns:
      cases:      list[str]                 e.g., ["Cases_20783_TP1", "Cases_20784_TP0", ...]
      modalities: dict[case_id] -> dict[ModalityKey] = /abs/path/to/file.mnc
                   e.g., modalities["Cases_20783_TP1"]["SWI"] = "/.../MS_Case_20783_TP1_SWI_7T.mnc"
    Grouping: study -> patient_id -> TP
    """

    # Excel in same folder as the caller (main training script)
    here = Path(__file__).resolve().parent
    xlsx_path = here / excel_name
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found next to script: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # robust column access (your sample headers)
    required_cols = ["file_path", "study", "patient_id", "modality", "TP", "Image Type", "filename"]
    for c in required_cols:
        if c not in df.columns:
            # tolerate 'filename' missing by synthesizing from file_path
            if c == "filename":
                df["filename"] = df["file_path"].map(lambda p: os.path.basename(str(p)))
                continue
            raise ValueError(f"Missing required column in Excel: '{c}'")

    # 1) filter by Image Type
    df["Image Type"] = df["Image Type"].astype(str)
    mask_cross = df["Image Type"].str.lower().eq(str(image_type_required).lower())
    df = df[mask_cross].copy()

    if df.empty:
        raise ValueError(f"No rows after filtering Image Type == '{image_type_required}'.")

    # normalize modalities
    df["mod_norm"] = df["modality"].map(_normalize_modality_name)

    # Optionally whitelist modalities
    if allowed_modalities:
        allowed_set = {m if m in {"T2","FLAIR","SWI","FLAIR-SWI","Paramagnetic","Diamagnetic","AMag","APhase"} else m
                       for m in allowed_modalities}
        df = df[df["mod_norm"].isin(allowed_set)].copy()

    # drop rows pointing to missing files (optional)
    if drop_missing_files:
        df["__exists"] = df["file_path"].map(lambda p: os.path.exists(str(p)))
        df = df[df["__exists"]].copy()
        df.drop(columns=["__exists"], inplace=True)

    if df.empty:
        raise ValueError("After filtering and file existence checks, no inputs remain.")

    # group by study / patient / TP (so each temporal point is its own 'case')
    cases = []
    modalities = {}

    for (study, pid, tp), g in df.groupby(["study", "patient_id", "TP"], dropna=False):
        case_id = f"{study}_{pid}_TP{int(tp) if pd.notna(tp) else 0}"
        # within this case, we may still have multiple rows per modality (multi-echo)
        per_mod_paths = {}

        for mod_name, gmod in g.groupby("mod_norm"):
            if len(gmod) == 1:
                chosen = gmod.iloc[0]
            else:
                chosen = _select_by_te(gmod, policy=te_policy, closest_ms=te_closest_ms)
            per_mod_paths[mod_name] = str(chosen["file_path"])

        if per_mod_paths:  # only keep cases with ≥1 modality
            cases.append(case_id)
            modalities[case_id] = per_mod_paths

    # Stable order
    cases = sorted(cases, key=lambda cid: (
        cid.split("_")[0],                      # study
        int(re.findall(r"_(\d+)_", cid)[0])     if re.search(r"_(\d+)_", cid) else 0,  # patient_id numeric if possible
        int(re.findall(r"TP(\d+)", cid)[0])     if re.search(r"TP(\d+)", cid) else 0   # TP
    ))

    return cases, modalities
