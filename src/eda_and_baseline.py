import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def read_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def add_features(df):
    now = pd.Timestamp.now()
    dob = pd.to_datetime(df["DATE_OF_BIRTH"], errors="coerce")
    reg = pd.to_datetime(df["REGISTRATION_DATE"], errors="coerce")

    age_days = (now - dob).dt.days
    df["AGE"] = age_days / 365.25

    tmp = df.groupby("ADDR_KEY")["SOS_VOTERID"].nunique().reset_index()
    tmp.columns = ["ADDR_KEY", "ADDR_DENSITY"]
    df = df.merge(tmp, on="ADDR_KEY", how="left")

    df["REG_DATE_ONLY"] = reg.dt.date
    return df


def make_eda(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    age = df["AGE"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(age, bins=80)
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "age_distribution.png"), dpi=150)
    plt.close()

    reg_cnt = df.groupby("REG_DATE_ONLY")["SOS_VOTERID"].count().reset_index()
    reg_cnt = reg_cnt.sort_values("REG_DATE_ONLY")
    plt.figure(figsize=(10, 5))
    plt.plot(reg_cnt["REG_DATE_ONLY"], reg_cnt["SOS_VOTERID"])
    plt.title("Registration timeline")
    plt.xlabel("Date")
    plt.ylabel("Registrations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "registration_timeline.png"), dpi=150)
    plt.close()


def zscore_flag(x, thr=3.0):
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return np.zeros(len(x))
    z = (x - m) / s
    return (np.abs(z) > thr).astype(int)


def iqr_flag(x, k=1.5):
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return ((x < low) | (x > high)).astype(int)


def run_baseline(df, contamination, seed):
    age = df["AGE"].astype(float).values
    den = df["ADDR_DENSITY"].astype(float).values

    a1 = zscore_flag(age, thr=3.0)
    a2 = iqr_flag(age, k=1.5)
    d1 = zscore_flag(den, thr=3.0)
    d2 = iqr_flag(den, k=1.5)

    rule_score = a1 + a2 + d1 + d2
    df["RULE_FLAG"] = (rule_score >= 1).astype(int)

    X = np.vstack([age, den]).T
    med_age = np.nanmedian(age)
    med_den = np.nanmedian(den)
    X[np.isnan(X[:, 0]), 0] = med_age
    X[np.isnan(X[:, 1]), 1] = med_den

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed
    )
    model.fit(X)
    score = model.decision_function(X)
    pred = model.predict(X)
    df["IF_SCORE"] = score
    df["IF_FLAG"] = (pred == -1).astype(int)

    mix = (df["RULE_FLAG"] + df["IF_FLAG"]).values
    df["FINAL_ANOMALY"] = (mix >= 1).astype(int)
    return df


def save_top(df, out_file, top_n):
    df2 = df[df["FINAL_ANOMALY"] == 1].copy()
    df2 = df2.sort_values("IF_SCORE", ascending=True)
    df3 = df2.head(top_n)
    df3.to_csv(out_file, index=False)
    print(f"[save] top anomalies: {out_file} rows={len(df3)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="output/swvf_processed.parquet")
    p.add_argument("--eda_dir", default="output/eda")
    p.add_argument("--out", default="output/anomalies_top500.csv")
    p.add_argument("--all_out", default="output/scored_all.csv")
    p.add_argument("--top_n", type=int, default=500)
    p.add_argument("--contamination", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=567)
    args = p.parse_args()

    df = read_df(args.input)
    df = add_features(df)
    make_eda(df, args.eda_dir)
    df = run_baseline(df, args.contamination, args.seed)
    df.to_csv(args.all_out, index=False)
    save_top(df, args.out, args.top_n)


if __name__ == "__main__":
    main()
