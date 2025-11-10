"""Data preprocessing helpers for hu-florido-v2."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import tomllib

# Root paths and filenames used by preprocessing
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config.toml"
MAPPING_FILENAME = "couples_gender_mapping.json"


def _norm_name(value: Optional[str]) -> str:
    """Normalize name for lookup."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _load_couples_gender_mapping(processed_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load couples/gender mapping from JSON file."""
    mapping_path = processed_dir / MAPPING_FILENAME
    if not mapping_path.exists() or mapping_path.stat().st_size == 0:
        default_mapping = {
            "Example Person": {
                "gender": "unknown",
                "couple": "single",
                "aliases": ["example-alias"],
            },
            "Unknown": {
                "gender": "unknown",
                "couple": "single",
                "aliases": ["unknown"],
            },
        }
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as mp:
            json.dump(default_mapping, mp, indent=4)
        print(
            f"⚠️  Created default mapping template at {mapping_path}. "
            "Please update it with your couples and aliases before rerunning."
        )

    try:
        with mapping_path.open("r", encoding="utf-8") as mp:
            raw_mapping = json.load(mp)
    except json.JSONDecodeError as exc:
        backup_path = mapping_path.with_suffix(".invalid.json")
        mapping_path.replace(backup_path)
        mapping_path.write_text("{}", encoding="utf-8")
        raise ValueError(
            f"Invalid JSON in {backup_path}. A blank template was created; "
            "please fix the file and rerun."
        ) from exc

    if not isinstance(raw_mapping, dict):
        raise ValueError("couples_gender_mapping.json must contain an object at the top level")

    real_lookup: Dict[str, Dict[str, Any]] = {}
    alias_lookup: Dict[str, Dict[str, Any]] = {}

    for name, info in raw_mapping.items():
        if not isinstance(info, dict):
            continue

        normalized_info = {
            "gender": str(info.get("gender", "unknown") or "unknown"),
            "couple": str(info.get("couple", "single") or "single"),
        }

        real_lookup[_norm_name(name)] = normalized_info

        aliases_field = info.get("aliases", [])
        if isinstance(aliases_field, str):
            aliases_field = [aliases_field]
        elif not isinstance(aliases_field, list):
            aliases_field = []

        for alias in aliases_field:
            if alias:
                alias_lookup[_norm_name(alias)] = normalized_info

    return {"real": real_lookup, "alias": alias_lookup}


def _lookup_mapping_field(
    name: Optional[str],
    field: str,
    mapping: Dict[str, Dict[str, Dict[str, Any]]],
) -> Optional[str]:
    """Look up gender or couple from mapping."""
    if not name:
        return None
    key = _norm_name(name)
    info = mapping["real"].get(key) or mapping["alias"].get(key)
    if info:
        value = info.get(field)
        if isinstance(value, str):
            return value
    return None


def run_preprocessing() -> Path:
    """
    Run the full preprocessing pipeline.
    Returns the path to the processed output CSV.
    """
    print("\n" + "=" * 70)
    print("RUNNING PREPROCESSING")
    print("=" * 70)

    # Load config
    with CONFIG_FILE.open("rb") as f:
        config = tomllib.load(f)

    # Build path to raw input file
    raw_dir = Path(config.get("raw", "data/raw")).resolve()
    input_file = config.get("input", "_chat.txt")
    datafile = raw_dir / input_file

    processed_dir = Path(config.get("processed", "data/processed")).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not datafile.exists():
        print(f"⚠️  Raw input file not found: {datafile}")
        print("   Looking for existing processed file instead...")
        return processed_dir / "output.csv"

    # Run wa_analyzer preprocessing
    try:
        import wa_analyzer.preprocess as preprocessor
        preprocessor.main(["--device", "old"], standalone_mode=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Preprocessor failed: {exc}")
        return processed_dir / "output.csv"

    output_csv = processed_dir / "output.csv"
    if not output_csv.exists():
        # Backward compatibility: fall back to timestamped file if present
        try:
            latest_csv = max(processed_dir.glob("whatsapp-*.csv"), key=lambda p: p.stat().st_mtime)
            output_csv = latest_csv
        except ValueError:
            return processed_dir / "output.csv"

    # Load data
    out_df = pd.read_csv(output_csv)

    # Preserve original author
    if "author" in out_df.columns and "author_orig" not in out_df.columns:
        out_df["author_orig"] = out_df["author"].astype(str)

    # Load mapping
    mapping_data = _load_couples_gender_mapping(processed_dir)

    # Add gender and couples columns
    if "author_orig" in out_df.columns:
        gender_series = out_df["author_orig"].apply(lambda name: _lookup_mapping_field(name, "gender", mapping_data))
        couples_series = out_df["author_orig"].apply(lambda name: _lookup_mapping_field(name, "couple", mapping_data))
    else:
        gender_series = pd.Series([None] * len(out_df), index=out_df.index, dtype=object)
        couples_series = pd.Series([None] * len(out_df), index=out_df.index, dtype=object)

    out_df["gender"] = gender_series.fillna("unknown").replace("", "unknown")
    out_df["couples"] = couples_series.fillna("single").replace("", "single")

    # Clean message text
    timestamp_pattern = r"^\s*\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*-\s*"
    author_pattern = r"^\s*\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*-\s*[^:]+:\s*"

    out_df["message"] = (
        out_df.get("message", pd.Series(dtype=str)).astype(str)
        .str.replace(author_pattern, "", regex=True)
        .str.replace(timestamp_pattern, "", regex=True)
        .str.replace(r"\s*\n+\s*$", "", regex=True)
        .str.replace(r"^\s+|\s+$", "", regex=True)
        .str.strip()
    )

    # Anonymization
    if "author" in out_df.columns:
        authors_before = set(out_df["author"].dropna().unique().tolist())

        ref_path = processed_dir / "anon_reference.json"
        if ref_path.exists():
            try:
                with ref_path.open("r", encoding="utf-8") as rf:
                    ref_raw = json.load(rf)

                def _norm(s: str) -> str:
                    return re.sub(r"\s+", " ", str(s).strip()).lower()

                candB_real_to_pseudo = {_norm(k): v for k, v in ref_raw.items()}
                authors_norm = {_norm(a) for a in authors_before}
                replB = sum(1 for a in authors_norm if a in candB_real_to_pseudo)

                if replB > 0:
                    out_df["author"] = [candB_real_to_pseudo.get(_norm(a), a) for a in out_df["author"].astype(str)]
            except Exception as exc:  # noqa: BLE001
                print(f"Anonymization failed: {exc}")
        else:
            adjectives = [
                "agile","brisk","calm","dapper","eager","fiery","gentle","hazy","icy","jaunty",
                "keen","lively","mellow","nimble","odd","plucky","quirky","rosy","spry","tidy",
                "urbane","vivid","witty","young","zesty","bold","clever","daring","elegant","frosty",
            ]
            animals = [
                "ant","beagle","cat","dolphin","eagle","falcon","gecko","heron","ibis","jaguar",
                "koala","llama","manatee","narwhal","otter","panda","quail","raccoon","seal","tiger",
                "urchin","viper","walrus","yak","zebra","goshawk","trout","magpie","jackal","duck",
            ]

            def make_pseudo(name: str) -> str:
                if not name or name.lower() == "unknown":
                    return "unknown"
                if "-" in name and name.count(" ") == 0:
                    return name.lower()
                hashed = hashlib.sha256(name.encode("utf-8")).hexdigest()
                adjective = adjectives[int(hashed[0:2], 16) % len(adjectives)]
                animal = animals[int(hashed[2:4], 16) % len(animals)]
                return f"{adjective}-{animal}"

            out_df["author"] = out_df["author"].astype(str).apply(make_pseudo)

    # Emoji detection
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    def has_emoji(text):
        if not isinstance(text, str) or not text:
            return False
        return bool(emoji_pattern.search(text))

    out_df["has_emoji"] = out_df["message"].apply(has_emoji)

    # Write output
    csv_out = processed_dir / "output.csv"
    parquet_path = processed_dir / "output.parq"

    if "author_orig" in out_df.columns:
        out_df_clean = out_df.drop(columns=["author_orig"])
    else:
        out_df_clean = out_df

    out_df_clean.to_csv(csv_out, index=False)
    try:
        out_df_clean.to_parquet(parquet_path, index=False)
    except Exception:  # noqa: BLE001
        pass

    # Remove intermediate files
    removed = 0
    for fp in processed_dir.glob("whatsapp-*.csv"):
        try:
            fp.unlink()
            removed += 1
        except OSError:
            pass

    print(f"✅ Cleaned & anonymized messages written to {csv_out}")
    print("=" * 70 + "\n")

    return csv_out


__all__ = ["run_preprocessing"]
