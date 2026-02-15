import argparse
import gzip
import os
import pandas as pd


def _detect_compression(filepath):
    """Check magic bytes to decide if a file is really gzip."""
    try:
        with open(filepath, "rb") as fh:
            magic = fh.read(2)
        if magic == b"\x1f\x8b":
            return "gzip"
    except Exception:
        pass
    return None


def _detect_sep(filepath, compression):
    """Read first line(s) to guess the delimiter."""
    try:
        if compression == "gzip":
            with gzip.open(filepath, "rt", errors="replace") as fh:
                first = fh.readline()
        else:
            with open(filepath, "r", errors="replace") as fh:
                first = fh.readline()
        if "|" in first:
            return "|"
        if "\t" in first:
            return "\t"
    except Exception:
        pass
    return ","


def read_all_files(file_list):
    all_df = []
    i = 0
    while i < len(file_list):
        f = file_list[i]
        print(f"[load] reading {f}")
        comp = _detect_compression(f)
        sep = _detect_sep(f, comp)
        print(f"[load]   compression={comp}  sep={repr(sep)}")
        df = pd.read_csv(f, dtype=str, low_memory=False, compression=comp, sep=sep)
        all_df.append(df)
        i += 1

    big = pd.concat(all_df, axis=0, ignore_index=True)
    print(f"[load] merged rows={len(big)} cols={len(big.columns)}")
    return big


def clean_text_col(df, col):
    if col not in df.columns:
        return df
    s = df[col].fillna("")
    s = s.astype(str)
    s = s.str.upper()
    s = s.str.strip()
    df[col] = s
    return df


def normalize_address(df):
    need = [
        "RESIDENTIAL_ADDRESS1",
        "RESIDENTIAL_SECONDARY_ADDR",
        "RESIDENTIAL_CITY",
        "RESIDENTIAL_STATE",
        "RESIDENTIAL_ZIP"
    ]

    i = 0
    while i < len(need):
        df = clean_text_col(df, need[i])
        i += 1

    a1 = df.get("RESIDENTIAL_ADDRESS1", "")
    a2 = df.get("RESIDENTIAL_SECONDARY_ADDR", "")
    city = df.get("RESIDENTIAL_CITY", "")
    state = df.get("RESIDENTIAL_STATE", "")
    z = df.get("RESIDENTIAL_ZIP", "")

    full = a1.astype(str) + " " + a2.astype(str) + ", " + city.astype(str) + ", " + state.astype(str) + " " + z.astype(str)
    full = full.str.replace(r"\s+", " ", regex=True)
    full = full.str.strip().str.upper()
    df["ADDR_KEY"] = full
    return df


def date_convert(df):
    d_cols = ["DATE_OF_BIRTH", "REGISTRATION_DATE"]
    i = 0
    while i < len(d_cols):
        c = d_cols[i]
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        i += 1
    return df


def basic_filter(df):
    keep = [
        "SOS_VOTERID", "COUNTY_NUMBER", "COUNTY_ID", "LAST_NAME", "FIRST_NAME",
        "DATE_OF_BIRTH", "REGISTRATION_DATE", "VOTER_STATUS",
        "RESIDENTIAL_ADDRESS1", "RESIDENTIAL_SECONDARY_ADDR", "RESIDENTIAL_CITY",
        "RESIDENTIAL_STATE", "RESIDENTIAL_ZIP", "ADDR_KEY"
    ]

    real_keep = []
    i = 0
    while i < len(keep):
        if keep[i] in df.columns:
            real_keep.append(keep[i])
        i += 1

    out = df[real_keep].copy()
    return out


def save_df(df, out_path):
    folder = os.path.dirname(out_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"[save] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out", default="output/swvf_processed.parquet")
    args = p.parse_args()

    df = read_all_files(args.files)
    df = normalize_address(df)
    df = date_convert(df)
    df = basic_filter(df)
    save_df(df, args.out)


if __name__ == "__main__":
    main()
