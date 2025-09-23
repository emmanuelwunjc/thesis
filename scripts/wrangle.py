import sys
import json
import subprocess
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

try:
    import pyreadr  # type: ignore
except Exception as e:  # pragma: no cover
    pyreadr = None

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    pd = None

DATA_PATH = Path(__file__).parent / "hints7_public copy.rda"


def ensure_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if missing:
        print(json.dumps({"error": "missing_deps", "packages": missing}))
        sys.exit(3)


def load_r_data_with_pyreadr(path: Path):
    if pyreadr is None:
        return None
    try:
        result = pyreadr.read_r(str(path))
    except Exception:
        return None
    for name, obj in result.items():
        if isinstance(obj, pd.DataFrame):
            return obj
    # fallback: try to coerce first object
    try:
        name, obj = next(iter(result.items()))
    except StopIteration:
        return None
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None


def load_r_data_with_rscript(path: Path) -> pd.DataFrame | None:
    # Use Rscript to load .rda via a temp .R script, then write CSV
    r_source = """
args <- commandArgs(trailingOnly=TRUE)
rda_path <- args[1]
out_csv <- args[2]
tryCatch({
  e <- new.env()
  ok <- file.exists(rda_path)
  if (!ok) stop(paste0("FILE_NOT_FOUND: ", rda_path))
  load(rda_path, envir=e)
  nms <- ls(envir=e)
  pick <- NULL
  if (length(nms) > 0) {
    for (nm in nms) {
      obj <- get(nm, envir=e)
      if (is.data.frame(obj) || inherits(obj, "tbl_df")) { pick <- obj; break }
    }
    if (is.null(pick)) {
      pick <- tryCatch(as.data.frame(get(nms[1], envir=e)), error=function(err) NULL)
    }
  }
  if (is.null(pick)) {
    write("NO_DATAFRAME", file=out_csv)
  } else {
    write.csv(pick, file=out_csv, row.names=FALSE, na="")
  }
}, error=function(err) {
  msg <- paste0("RSCRIPT_ERROR: ", conditionMessage(err))
  write(msg, file=out_csv)
})
"""
    from shutil import which
    if which("Rscript") is None:
        return None
    with TemporaryDirectory() as td:
        out_csv = Path(td) / "export.csv"
        r_file = Path(td) / "export.R"
        try:
            r_file.write_text(r_source, encoding="utf-8")
        except Exception:
            return None
        try:
            proc = subprocess.run(
                ["Rscript", str(r_file), str(path), str(out_csv)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return None
        except subprocess.CalledProcessError as e:
            diag = {"error": "rscript_failed", "returncode": e.returncode, "stdout": e.stdout[-400:], "stderr": e.stderr[-400:]}
            print(json.dumps(diag))
            return None
        try:
            head = out_csv.read_text(encoding="utf-8", errors="ignore")[:200]
            if head.startswith("NO_DATAFRAME") or head.startswith("RSCRIPT_ERROR:"):
                print(json.dumps({"error": "rscript_diag", "head": head}))
                return None
        except Exception:
            return None
        try:
            df = pd.read_csv(out_csv)
        except Exception:
            return None
        return df


def load_r_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(json.dumps({"error": "missing_file", "path": str(path)}))
        sys.exit(2)
    # Try native pyreadr first
    df = load_r_data_with_pyreadr(path)
    if df is not None:
        return df
    # Fallback to Rscript
    df = load_r_data_with_rscript(path)
    if df is not None:
        return df
    print(json.dumps({"error": "unable_to_load_rda", "path": str(path)}))
    sys.exit(4)


def detect_diabetes_and_categories(df: 'pd.DataFrame') -> dict:
    columns_lower = {c.lower(): c for c in df.columns}
    candidates = [
        # direct self-reported condition
        'medconditions_diabetes',
        # medication markers (if present)
        'diabetes_meds', 'diabetes_medications', 'insulin', 'takesinsulin',
        # diagnosis flag variants
        'diabetes', 'diabetes_dx', 'diabetesdiagnosed',
        # a1c category (if present)
        'a1c', 'a1c_cat', 'hba1c',
    ]
    available = [columns_lower[c] for c in candidates if c in columns_lower]

    result = {"available_fields": available}

    # Heuristic primary indicator
    diabetic_mask = None
    reasons = {}

    def or_mask(m1, m2):
        if m1 is None:
            return m2
        if m2 is None:
            return m1
        return m1 | m2

    # Helper to normalize yes/no like variables
    def yes_mask(series: 'pd.Series'):
        s = series.astype(str).str.strip().str.lower()
        return s.isin(["1", "y", "yes", "true", "t", "checked", "diabetic", "diagnosed", "currently have", "have"])

    def numeric_positive(series: 'pd.Series'):
        s = pd.to_numeric(series, errors='coerce')
        return s.notna() & (s > 0)

    # 1) Direct self-reported condition
    col = columns_lower.get('medconditions_diabetes')
    if col in df.columns:
        m = yes_mask(df[col]) | numeric_positive(df[col])
        diabetic_mask = or_mask(diabetic_mask, m)
        reasons['medconditions_diabetes'] = int(m.sum())

    # 2) Generic diabetes flags
    for key in ['diabetes', 'diabetes_dx', 'diabetesdiagnosed']:
        col = columns_lower.get(key)
        if col in df.columns:
            m = yes_mask(df[col]) | numeric_positive(df[col])
            diabetic_mask = or_mask(diabetic_mask, m)
            reasons[key] = int(m.sum())

    # 3) Medication-related (insulin etc.)
    for key in ['diabetes_meds', 'diabetes_medications', 'insulin', 'takesinsulin']:
        col = columns_lower.get(key)
        if col in df.columns:
            m = yes_mask(df[col]) | numeric_positive(df[col])
            diabetic_mask = or_mask(diabetic_mask, m)
            reasons[key] = int(m.sum())

    # 4) A1C heuristic: HbA1c >= 6.5 indicates diabetes (if available)
    for key in ['a1c', 'hba1c']:
        col = columns_lower.get(key)
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            m = s >= 6.5
            diabetic_mask = or_mask(diabetic_mask, m)
            reasons[key + '>=6.5'] = int(m.sum())

    if diabetic_mask is None:
        diabetic_mask = pd.Series(False, index=df.index)

    # Categorize: by treatment/medication if possible, else unknown
    categories = pd.Series('Unknown', index=df.index)

    insulin_cols = [columns_lower[k] for k in ['insulin', 'takesinsulin'] if k in columns_lower]
    med_cols = [columns_lower[k] for k in ['diabetes_meds', 'diabetes_medications'] if k in columns_lower]

    insulin_mask = None
    for c in insulin_cols:
        m = yes_mask(df[c]) | numeric_positive(df[c])
        insulin_mask = or_mask(insulin_mask, m)
    if insulin_mask is None:
        insulin_mask = pd.Series(False, index=df.index)

    meds_mask = None
    for c in med_cols:
        m = yes_mask(df[c]) | numeric_positive(df[c])
        meds_mask = or_mask(meds_mask, m)
    if meds_mask is None:
        meds_mask = pd.Series(False, index=df.index)

    categories = categories.mask(diabetic_mask & insulin_mask, 'Insulin-treated')
    categories = categories.mask(diabetic_mask & (~insulin_mask) & meds_mask, 'Non-insulin meds')
    categories = categories.mask(diabetic_mask & (~insulin_mask) & (~meds_mask), 'No meds reported')

    summary = {
        "num_diabetic": int((diabetic_mask).sum()),
        "num_non_diabetic": int((~diabetic_mask).sum()),
        "by_category": categories[diabetic_mask].value_counts(dropna=False).to_dict(),
        "reasons_counts": reasons,
    }
    return summary


def derive_diabetes_mask(df: 'pd.DataFrame') -> 'pd.Series':
    columns_lower = {c.lower(): c for c in df.columns}
    def or_mask(m1, m2):
        if m1 is None:
            return m2
        if m2 is None:
            return m1
        return m1 | m2
    def yes_mask(series: 'pd.Series'):
        s = series.astype(str).str.strip().str.lower()
        return s.isin(["1", "y", "yes", "true", "t", "checked", "diabetic", "diagnosed", "currently have", "have"])
    def numeric_positive(series: 'pd.Series'):
        s = pd.to_numeric(series, errors='coerce')
        return s.notna() & (s > 0)
    diabetic_mask = None
    for key in ['medconditions_diabetes', 'diabetes', 'diabetes_dx', 'diabetesdiagnosed']:
        col = columns_lower.get(key)
        if col in df.columns:
            m = yes_mask(df[col]) | numeric_positive(df[col])
            diabetic_mask = or_mask(diabetic_mask, m)
    if diabetic_mask is None:
        diabetic_mask = pd.Series(False, index=df.index)
    return diabetic_mask


def guess_demographic_columns(df: 'pd.DataFrame') -> dict:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    patterns = {
        'age': ['age', 'age_cat', 'agegroup', 'agegrp'],
        'sex': ['sex', 'gender'],
        'race': ['race', 'race_ethnicity', 'racecat', 'race_cat'],
        'hispanic': ['hispanic', 'latino'],
        'education': ['educ', 'education', 'education_level', 'educ_cat'],
        'income': ['income', 'income_cat', 'hhincome'],
        'marital': ['marital', 'marital_status'],
        'employment': ['employ', 'employment', 'workstatus'],
        'insurance': ['insurance', 'healthinsurance', 'insured'],
        'region': ['region', 'censreg', 'censdiv', 'app_region'],
        'urban_rural': ['ruc2003', 'ruc2013', 'ruc2023', 'ruca', 'nchs', 'nchsurcode2013'],
    }
    found: dict[str, str] = {}
    for key, pats in patterns.items():
        for p in pats:
            if p in lower_map:
                found[key] = lower_map[p]
                break
    return found


def pick_privacy_related_columns(df: 'pd.DataFrame') -> list[str]:
    cols = list(df.columns)
    # Heuristic: variables about records, sharing, portals, trust, confidence, privacy
    keywords = [
        'privacy', 'share', 'shared', 'records', 'record', 'portal', 'online', 'data', 'deviceinfo', 'device',
        'trust', 'confident', 'confidence', 'willing', 'labshare', 'accessonlinerecord', 'understandonlinemedrec',
        'hcpencourageonlinerec', 'misleadinghealthinfo'
    ]
    cols_lower = [c.lower() for c in cols]
    selected = []
    for c, cl in zip(cols, cols_lower):
        if any(k in cl for k in keywords):
            selected.append(c)
    # Deduplicate and return a manageable subset (cap to 60 to avoid noise)
    return selected[:60]


def crosstab_counts(df: 'pd.DataFrame', mask: 'pd.Series', column: str) -> dict:
    # Returns JSON-serializable counts and percentages
    temp = pd.DataFrame({
        'diabetic': mask.map({True: 'Diabetic', False: 'Non-Diabetic'}),
        column: df[column].astype(str).fillna('NA')
    })
    ct = pd.crosstab(temp['diabetic'], temp[column], dropna=False)
    pct = ct.div(ct.sum(axis=0), axis=1).round(4)
    return {
        'count': ct.to_dict(),
        'percent_by_col': pct.to_dict(),
    }


def analyze_privacy(df: 'pd.DataFrame', mask: 'pd.Series', columns: list[str]) -> dict:
    analysis = {}
    for col in columns:
        try:
            analysis[col] = crosstab_counts(df, mask, col)
        except Exception:
            continue
    return analysis


def get_age_series(df: 'pd.DataFrame', age_col: str) -> 'pd.Series':
    s = pd.to_numeric(df[age_col], errors='coerce')
    return s


def run_age_band_analysis(df: 'pd.DataFrame', age_col: str, age_min: int, age_max: int, label: str) -> dict:
    ages = get_age_series(df, age_col)
    band_mask = (ages >= age_min) & (ages <= age_max)
    df_band = df.loc[band_mask].copy()
    dia_mask_band = derive_diabetes_mask(df_band)

    # Summary within band
    total = int(df_band.shape[0])
    num_dia = int(dia_mask_band.sum())
    num_non = total - num_dia

    # Age distribution within band (percent by age value)
    age_counts = ages[band_mask].value_counts().sort_index()
    age_pct = (age_counts / age_counts.sum() * 100).round(2).to_dict()

    # Privacy analysis within band
    privacy_cols = pick_privacy_related_columns(df_band)
    privacy = analyze_privacy(df_band, dia_mask_band, privacy_cols)

    out = {
        "label": label,
        "age_min": age_min,
        "age_max": age_max,
        "n": total,
        "num_diabetic": num_dia,
        "num_non_diabetic": num_non,
        "age_percent_distribution": age_pct,
        "privacy_columns": privacy_cols,
        "privacy_analysis": privacy,
    }
    return out


def detect_weights(df: 'pd.DataFrame') -> dict:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    main_weight = None
    # Prefer typical names
    for key in ['finalwt', 'final_wt', 'weight', 'wt', 'sampleweight', 'samplingweight']:
        if key in lower_map:
            main_weight = lower_map[key]
            break
    if main_weight is None:
        # fallback: first column containing 'weight' substring
        for c in cols:
            if 'weight' in c.lower() or c.lower().endswith('wt'):
                main_weight = c
                break
    # Replicate weights
    replicate_weights: list[str] = []
    for c in cols:
        cl = c.lower()
        if ('rep' in cl or 'jack' in cl or 'brr' in cl) and ('weight' in cl or cl.endswith('wt')):
            replicate_weights.append(c)
    # Sort replicates numerically if numbered
    def rep_key(name: str):
        import re
        m = re.search(r'(\d+)$', name)
        return int(m.group(1)) if m else 0
    replicate_weights.sort(key=rep_key)
    return {
        'main': main_weight,
        'replicates': replicate_weights
    }


def build_dummies_for_column(series: 'pd.Series', max_levels: int = 6) -> dict:
    s = series.fillna('NA').astype(str).str.strip()
    # Numeric binary
    numeric = pd.to_numeric(s, errors='coerce')
    if numeric.notna().any():
        # If values mostly 0/1
        vals = numeric.dropna().unique()
        if set(pd.Series(vals).dropna().astype(int).unique()).issubset({0,1}):
            return {'_binary': (numeric.fillna(0) > 0).astype(int)}
    # Categorical levels: create up to max_levels one-hot dummies
    levels = s.value_counts().index.tolist()[:max_levels]
    dummies = {}
    for lvl in levels:
        dummies[f"{lvl}"] = (s == lvl).astype(int)
    return dummies


def compute_weighted_mean(x: 'pd.Series', w: Optional['pd.Series']) -> float:
    x = pd.to_numeric(x, errors='coerce')
    if w is None:
        return float(x.mean(skipna=True))
    w = pd.to_numeric(w, errors='coerce')
    mask = x.notna() & w.notna()
    if mask.sum() == 0:
        return float('nan')
    return float((x[mask] * w[mask]).sum() / w[mask].sum())


def jackknife_se(est_full: float, est_reps: list[float]) -> float:
    # Delete-1 jackknife standard error: sqrt(((R-1)/R) * sum((theta_r - theta_full)^2))
    import math
    R = len(est_reps)
    if R == 0 or est_full != est_full:
        return float('nan')
    s = sum((er - est_full) * (er - est_full) for er in est_reps)
    return float(math.sqrt(((R - 1.0) / R) * s))


def compare_privacy_dummies(df: 'pd.DataFrame', dia_mask: 'pd.Series', privacy_cols: list[str], weights: dict) -> dict:
    results = {}
    main_w = weights.get('main')
    rep_ws = weights.get('replicates', [])
    for col in privacy_cols:
        try:
            dummies = build_dummies_for_column(df[col])
        except Exception:
            continue
        col_result = {}
        for dummy_name, dummy_series in dummies.items():
            # Group means
            mean_dia = compute_weighted_mean(dummy_series[dia_mask], df[main_w] if main_w else None)
            mean_non = compute_weighted_mean(dummy_series[~dia_mask], df[main_w] if main_w else None)
            diff = mean_dia - mean_non if (mean_dia == mean_dia and mean_non == mean_non) else float('nan')
            # Replicate diffs for SE
            rep_diffs: list[float] = []
            for rw in rep_ws:
                try:
                    md = compute_weighted_mean(dummy_series[dia_mask], df[rw])
                    mn = compute_weighted_mean(dummy_series[~dia_mask], df[rw])
                    rep_diffs.append(md - mn)
                except Exception:
                    continue
            se = jackknife_se(diff, rep_diffs) if rep_diffs else None
            col_result[dummy_name] = {
                'mean_diabetic': mean_dia,
                'mean_non_diabetic': mean_non,
                'diff': diff,
                'se_jackknife': se,
            }
        if col_result:
            results[col] = col_result
    return results


def main() -> None:
    ensure_dependencies()
    df = load_r_data(DATA_PATH)
    info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns[:200]),
    }
    summary = detect_diabetes_and_categories(df)
    dia_mask = derive_diabetes_mask(df)

    # Demographics
    demo = guess_demographic_columns(df)
    crosstabs = {}
    for key, col in demo.items():
        try:
            ct = crosstab_counts(df, dia_mask, col)
            crosstabs[key] = ct
        except Exception:
            continue

    # Privacy-related variables
    privacy_cols = pick_privacy_related_columns(df)
    privacy_analysis = analyze_privacy(df, dia_mask, privacy_cols)

    out_dir = Path(__file__).parent
    (out_dir / "diabetes_summary.json").write_text(
        json.dumps({"info": info, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "diabetes_demographics_crosstabs.json").write_text(
        json.dumps({"demographics": demo, "crosstabs": crosstabs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "diabetes_privacy_analysis.json").write_text(
        json.dumps({"privacy_columns": privacy_cols, "analysis": privacy_analysis}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Optional age-band analyses via CLI
    args = sys.argv[1:]
    if "--age-band" in args or "--age-iqr" in args:
        age_col = demo.get('age') or 'Age'
        ages = get_age_series(df, age_col)
        # Clean plausible ages for IQR calc
        plausible = ages[(ages >= 18) & (ages <= 100)].dropna()
        bands: list[tuple[int,int,str]] = []
        if "--age-band" in args:
            idx = args.index("--age-band")
            try:
                a_min = int(args[idx+1]); a_max = int(args[idx+2])
                bands.append((a_min, a_max, f"fixed_{a_min}_{a_max}"))
            except Exception:
                pass
        if "--age-iqr" in args and not plausible.empty:
            q1 = int(plausible.quantile(0.25))
            q3 = int(plausible.quantile(0.75))
            bands.append((q1, q3, f"iqr_{q1}_{q3}"))
        results = {}
        for a_min, a_max, label in bands:
            results[label] = run_age_band_analysis(df, age_col, a_min, a_max, label)
        if results:
            (out_dir / "age_band_analyses.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Optional weighted privacy dummy comparisons
    if "--privacy-dummies" in args:
        weights = detect_weights(df)
        diamsk = dia_mask
        priv_cols = privacy_cols
        comp = compare_privacy_dummies(df, diamsk, priv_cols, weights)
        (out_dir / "privacy_dummies_compare.json").write_text(
            json.dumps({
                "weights": weights,
                "variables": priv_cols,
                "comparison": comp
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps({
        "ok": True,
        "info": info,
        "summary": summary,
        "demographics": demo,
        "num_privacy_columns": len(privacy_cols)
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
