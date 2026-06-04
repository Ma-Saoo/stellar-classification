"""
fetch_sdss.py
-------------
Module for fetching and preparing SDSS stellar data.
Import this from Star_Classification.py — do not run directly.

Functions:
    fetch_stars(n)         -- query SDSS SkyServer, return raw DataFrame
    clean_and_engineer(df) -- clean labels, add color indices
    summarize(df)          -- print dataset summary to console
    save(df, out_dir)      -- write raw + ML-ready CSVs to disk

Label parsing strategy:
    Raw SDSS subClass strings are messy — they mix spectral type with
    luminosity class suffixes (e.g. "G2V", "B0.5Iae", "K1III").
    We extract just the letter + integer subtype number:

        "G5"       -> spectral_subclass="G5",  spectral_class="G"
        "G2V"      -> spectral_subclass="G2",  spectral_class="G"
        "B0.5Iae"  -> spectral_subclass="B0",  spectral_class="B"
        "K1III"    -> spectral_subclass="K1",  spectral_class="K"
        "WD"       -> rejected
        "sdF3"     -> rejected
"""

import os
import re
import time

import pandas as pd
from astroquery.sdss import SDSS

MAIN_SEQUENCE_TYPES = set("OBAFGKM")

FEATURE_COLS = ["u", "g", "r", "i", "z",
                "u_g", "g_r", "r_i", "i_z", "u_r", "g_i",
                "spectral_subclass", "spectral_class"]

SDSS_QUERY = """
SELECT TOP {n}
    s.specobjid,
    s.subClass,
    p.objID,
    p.ra,
    p.dec,
    p.u, p.g, p.r, p.i, p.z,
    p.err_u, p.err_g, p.err_r, p.err_i, p.err_z,
    s.elodieTEff,
    s.elodieLogG,
    s.elodieFeH,
    s.elodieZ,
    s.snMedian
FROM SpecObj AS s
JOIN PhotoObj AS p ON s.bestObjID = p.objID
WHERE
    s.class = 'STAR'
    AND s.subClass != ''
    AND s.subClass NOT LIKE '%:%'
    AND p.u BETWEEN 14 AND 23
    AND p.g BETWEEN 14 AND 23
    AND p.r BETWEEN 14 AND 23
    AND p.i BETWEEN 14 AND 23
    AND p.z BETWEEN 14 AND 23
    AND p.err_u < 0.2
    AND p.err_g < 0.2
    AND p.err_r < 0.2
    AND p.mode = 1
"""


def parse_spectral_subclass(subclass: str) -> tuple[str | None, str | None]:
    """
    Returns (coarse_letter, clean_subtype) or (None, None) if rejected.
    Strips luminosity suffixes, rounds decimals, rejects non-main-sequence.
    """
    if not isinstance(subclass, str):
        return None, None
    s = subclass.strip().upper()
    for prefix in ("SD", "WD", "CV", "C ", "C+", "CARBON", "SUBDWARF"):
        if s.startswith(prefix):
            return None, None
    if not s or s[0] not in MAIN_SEQUENCE_TYPES:
        return None, None
    match = re.match(r"([OBAFGKM])(\d+)?", s)
    if not match:
        return None, None
    letter = match.group(1)
    number = match.group(2) or ""
    return letter, f"{letter}{number}"


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]
    return df


def fetch_stars(n: int = 100_000) -> pd.DataFrame:
    print(f"[1/4] Querying SDSS SkyServer for up to {n:,} stars...")
    print("      (this usually takes 30–120 seconds)\n")
    t0     = time.time()
    result = SDSS.query_sql(SDSS_QUERY.format(n=n), timeout=300)
    print(f"      Query returned {len(result):,} rows in {time.time()-t0:.1f}s\n")
    return result.to_pandas()


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse labels and engineer color features.

    spectral_subclass  -- clean granular label e.g. "G2"  (ML training target)
    spectral_class     -- coarse letter e.g. "G"          (subclass[0], eval label)
    """
    print("[2/4] Cleaning labels and engineering features...")
    parsed = df["subClass"].apply(parse_spectral_subclass)
    df = df.copy()
    df["spectral_class"]    = parsed.apply(lambda x: x[0])
    df["spectral_subclass"] = parsed.apply(lambda x: x[1])

    before = len(df)
    df = df.dropna(subset=["spectral_subclass"])
    print(f"      Dropped {before - len(df):,} rows with non-standard labels")
    print(f"      Remaining: {len(df):,} stars\n")

    df = df.dropna(subset=["u", "g", "r", "i", "z"])
    df = add_color_indices(df)
    return df


def summarize(df: pd.DataFrame) -> None:
    print("[3/4] Dataset summary")
    print("=" * 45)
    counts = df["spectral_class"].value_counts().sort_index()
    total  = len(df)
    print(f"{'Class':<8} {'Count':>8} {'%':>7}")
    print("-" * 25)
    for cls, cnt in counts.items():
        print(f"{cls:<8} {cnt:>8,} {cnt/total*100:>6.1f}%")
    print("-" * 25)
    print(f"{'TOTAL':<8} {total:>8,}\n")

    n_sub = df["spectral_subclass"].nunique()
    print(f"Unique spectral subtypes (ML training targets): {n_sub}")
    sub_counts = df["spectral_subclass"].value_counts().sort_index()
    print(f"{'Subclass':<12} {'Count':>8}")
    print("-" * 22)
    for sub, cnt in sub_counts.items():
        print(f"{sub:<12} {cnt:>8,}")
    print()

    print("Color index statistics (mean ± std):")
    for col in ["u_g", "g_r", "r_i", "i_z"]:
        m, s = df[col].mean(), df[col].std()
        print(f"  {col}: {m:+.3f} ± {s:.3f}")
    print()

    if "elodieTEff" in df.columns:
        valid_t = df["elodieTEff"].replace(0, pd.NA).dropna()
        print(f"T_eff available for {len(valid_t):,} / {total:,} stars "
              f"(range: {valid_t.min():.0f}–{valid_t.max():.0f} K)\n")


def save(df: pd.DataFrame, out_dir: str) -> None:
    print(f"[4/4] Saving to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    full_path = os.path.join(out_dir, "sdss_stars_raw.csv")
    df.to_csv(full_path, index=False)
    print(f"      sdss_stars_raw.csv  ({len(df):,} rows, all columns)")

    ml_cols = [c for c in FEATURE_COLS if c in df.columns]
    ml_path = os.path.join(out_dir, "sdss_stars_ml.csv")
    df[ml_cols].to_csv(ml_path, index=False)
    print(f"      sdss_stars_ml.csv   ({len(df):,} rows)")
    print(f"        columns: {ml_cols}\n")
