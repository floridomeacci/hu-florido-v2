import wa_analyzer.preprocess as preprocessor
from pathlib import Path
import tomllib
from loguru import logger
import pandas as pd
import re
import json
import os
import hashlib

configfile = Path("config.toml").resolve()

def main():
    print("Running preprocessing...")
    with configfile.open("rb") as f:
        config = tomllib.load(f)

    datafile = Path(config.get("inputpath", "")).resolve()
    if not datafile.exists():
        logger.warning(f"{datafile} does not exist. Please check the path in config.toml.")
    else:
        try:
            logger.info(pd.read_csv(datafile, nrows=3).head())
        except Exception:
            pass

    try:
        preprocessor.main(["--device", "old"], standalone_mode=False)
    except Exception as e:
        logger.error(f"Preprocessor failed: {e}")
        return

    processed_dir = Path(config.get("processed", "data/processed")).resolve()
    try:
        latest_csv = max(processed_dir.glob("whatsapp-*.csv"), key=lambda p: p.stat().st_mtime)
    except ValueError:
        logger.warning(f"No whatsapp-*.csv found in {processed_dir}; skipping clean")
        return

    out_df = pd.read_csv(latest_csv)
    if "author" in out_df.columns and "author_orig" not in out_df.columns:
        out_df["author_orig"] = out_df["author"].astype(str)

    gender_mapping = {
        "Fei": "male",
        "Shuyan": "female",
        "Ramunė Meacci": "female", 
        "Florido Meacci": "male",
        "Alberto Doncato": "male",
        "Albert Doncato": "male",
        "Ludovica": "female",
        "Unknown": "unknown"
    }
    
    out_df["gender"] = out_df["author_orig"].map(gender_mapping).fillna("unknown")
    logger.info(f"Added gender column: {out_df['gender'].value_counts().to_dict()}")

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

    if "author" not in out_df.columns:
        logger.warning("'author' column missing; cannot anonymize")
    else:
        authors_before = set(out_df["author"].dropna().unique().tolist())
        logger.info(f"Unique authors BEFORE mapping ({len(authors_before)}): {sorted(list(authors_before))}")

        ref_path = processed_dir / "anon_reference.json"
        ref_raw = None
        if ref_path.exists():
            try:
                with ref_path.open("r", encoding="utf-8") as rf:
                    ref_raw = json.load(rf)
            except Exception as e:
                logger.warning(f"Failed to read anon_reference.json: {e}")

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", str(s).strip()).lower()

        if ref_raw:
            candA_real_to_pseudo = { _norm(v): k for k, v in ref_raw.items() }
            candB_real_to_pseudo = { _norm(k): v for k, v in ref_raw.items() }

            authors_norm = {_norm(a) for a in authors_before}
            replA = sum(1 for a in authors_norm if a in candA_real_to_pseudo)
            replB = sum(1 for a in authors_norm if a in candB_real_to_pseudo)

            if replB >= replA and replB > 0:
                chosen = candB_real_to_pseudo
                direction = "real->pseudo (keys real)"
            elif replA > 0:
                chosen = candA_real_to_pseudo
                direction = "real->pseudo (values real)"
            else:
                chosen = {}
                direction = "no_match"

            if chosen:
                out_df["author"] = [ chosen.get(_norm(a), a) for a in out_df["author"].astype(str) ]
                authors_after = set(out_df["author"].dropna().unique().tolist())
                newly = len({a for a in authors_after if a not in authors_before})
                logger.info(f"Applied anonymization mapping direction={direction}; replaced {newly} authors")
            else:
                logger.warning("anon_reference.json provided but no author strings matched; check casing / contents")
        else:
            logger.warning("anon_reference.json not found; applying fallback pseudonymization")
            adjectives = [
                "agile","brisk","calm","dapper","eager","fiery","gentle","hazy","icy","jaunty",
                "keen","lively","mellow","nimble","odd","plucky","quirky","rosy","spry","tidy",
                "urbane","vivid","witty","young","zesty","bold","clever","daring","elegant","frosty"
            ]
            animals = [
                "ant","beagle","cat","dolphin","eagle","falcon","gecko","heron","ibis","jaguar",
                "koala","llama","manatee","narwhal","otter","panda","quail","raccoon","seal","tiger",
                "urchin","viper","walrus","yak","zebra","goshawk","trout","magpie","jackal","duck"
            ]

            def make_pseudo(name: str) -> str:
                if not name or name.lower() == "unknown":
                    return "unknown"
                if "-" in name and name.count(" ") == 0:
                    return name.lower()
                h = hashlib.sha256(name.encode("utf-8")).hexdigest()
                adj = adjectives[int(h[0:2], 16) % len(adjectives)]
                ani = animals[int(h[2:4], 16) % len(animals)]
                return f"{adj}-{ani}"

            out_df["author"] = out_df["author"].astype(str).apply(make_pseudo)
            authors_after = set(out_df["author"].dropna().unique().tolist())
            logger.info(f"Fallback pseudonymization produced {len(authors_after)} unique pseudonyms")

    couples_map = {}
    couples_ref_path = processed_dir / "couples_reference.json"
    if couples_ref_path.exists():
        try:
            with couples_ref_path.open("r", encoding="utf-8") as cf:
                couples_ref = json.load(cf)
            if "author_orig" in out_df.columns:
                real_matches = sum(out_df["author_orig"].isin(couples_ref.keys()))
            else:
                real_matches = 0
            pseudo_matches = sum(out_df.get("author", pd.Series()).isin(couples_ref.keys()))
            if real_matches >= pseudo_matches and real_matches > 0 and "author_orig" in out_df.columns:
                out_df["couples"] = out_df["author_orig"].map(couples_ref)
                logger.info(f"Applied couples mapping using author_orig (matched {real_matches})")
            elif pseudo_matches > 0:
                out_df["couples"] = out_df["author"].map(couples_ref)
                logger.info(f"Applied couples mapping using pseudonyms (matched {pseudo_matches})")
            else:
                logger.warning("couples_reference.json present but no names matched; leaving as single")
        except Exception as e:
            logger.warning(f"Failed to read couples_reference.json: {e}")
    if "couples" in out_df.columns:
        out_df["couples"] = out_df["couples"].fillna("single")
    else:
        out_df["couples"] = "single"

    out_df["has_emoji"] = out_df["message"].apply(has_emoji)

    csv_out = processed_dir / "output.csv"
    parquet_path = processed_dir / "output.parq"
    out_df.to_csv(csv_out, index=False)
    try:
        out_df.to_parquet(parquet_path, index=False)
    except Exception as e:
        logger.warning(f"Could not write parquet file: {e}")

    if "author_orig" in out_df.columns:
        out_df_no_orig = out_df.drop(columns=["author_orig"])
        out_df_no_orig.to_csv(csv_out, index=False)
        try:
            out_df_no_orig.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    removed = 0
    for fp in processed_dir.glob("whatsapp-*.csv"):
        try:
            fp.unlink()
            removed += 1
        except OSError:
            pass
    logger.info(f"Removed {removed} intermediate whatsapp-*.csv file(s)")

    logger.success(f"Cleaned & anonymized messages written to {csv_out} and {parquet_path}")
        

if __name__ == "__main__":
    main()