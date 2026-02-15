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

    # address density
    tmp = df.groupby("ADDR_KEY")["SOS_VOTERID"].nunique().reset_index()
    tmp.columns = ["ADDR_KEY", "ADDR_DENSITY"]
    df = df.merge(tmp, on="ADDR_KEY", how="left")

    df["REG_DATE_ONLY"] = reg.dt.date

    # registration velocity -- how many ppl registered same day
    vel = df.groupby("REG_DATE_ONLY")["SOS_VOTERID"].count().reset_index()
    vel.columns = ["REG_DATE_ONLY", "REG_VELOCITY"]
    df = df.merge(vel, on="REG_DATE_ONLY", how="left")
    df["REG_VELOCITY"] = df["REG_VELOCITY"].fillna(0).astype(float)

    # duplicate entity detection: same last+first+dob = suspicious
    df["_name_dob"] = df["LAST_NAME"].astype(str) + "_" + df["FIRST_NAME"].astype(str) + "_" + df["DATE_OF_BIRTH"].astype(str)
    dup = df.groupby("_name_dob")["SOS_VOTERID"].count().reset_index()
    dup.columns = ["_name_dob", "DUP_COUNT"]
    df = df.merge(dup, on="_name_dob", how="left")
    df["DUP_FLAG"] = (df["DUP_COUNT"] > 1).astype(int)
    df.drop(columns=["_name_dob"], inplace=True)

    return df


def make_eda(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1 age distribution
    age = df["AGE"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(age, bins=80)
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "age_distribution.png"), dpi=150)
    plt.close()

    # 2 registration timeline
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

    # 3 address density distribution (log scale helps)
    den = df["ADDR_DENSITY"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(den, bins=100, log=True)
    plt.title("Address density distribution (log y)")
    plt.xlabel("Voters per address")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "addr_density_dist.png"), dpi=150)
    plt.close()

    # 4 county level voter count
    if "COUNTY_NUMBER" in df.columns:
        county_cnt = df.groupby("COUNTY_NUMBER")["SOS_VOTERID"].count().reset_index()
        county_cnt.columns = ["COUNTY_NUMBER", "VOTER_COUNT"]
        county_cnt = county_cnt.sort_values("VOTER_COUNT", ascending=False)
        plt.figure(figsize=(14, 5))
        plt.bar(county_cnt["COUNTY_NUMBER"].astype(str), county_cnt["VOTER_COUNT"])
        plt.title("Voter count by county")
        plt.xlabel("County number")
        plt.ylabel("Count")
        plt.xticks(rotation=90, fontsize=5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "county_voter_count.png"), dpi=150)
        plt.close()

    # 5 top high density addresses
    top_addr = df.drop_duplicates(subset=["ADDR_KEY"])[["ADDR_KEY", "ADDR_DENSITY"]].copy()
    top_addr = top_addr.sort_values("ADDR_DENSITY", ascending=False).head(50)
    top_addr.to_csv(os.path.join(out_dir, "top50_high_density_addresses.csv"), index=False)
    print(f"[eda] saved top50 high density addresses")

    # 6 age box plot by voter status
    if "VOTER_STATUS" in df.columns:
        status_vals = df["VOTER_STATUS"].dropna().unique().tolist()
        box_data = []
        box_labels = []
        i = 0
        while i < len(status_vals):
            s = status_vals[i]
            vals = df[df["VOTER_STATUS"] == s]["AGE"].dropna().values
            if len(vals) > 0:
                box_data.append(vals)
                box_labels.append(s)
            i += 1
        if len(box_data) > 0:
            plt.figure(figsize=(8, 5))
            plt.boxplot(box_data, labels=box_labels, showfliers=False)
            plt.title("Age by voter status")
            plt.ylabel("Age")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "age_boxplot_by_status.png"), dpi=150)
            plt.close()

    # 7 registration velocity distribution (spike detection viz)
    if "REG_VELOCITY" in df.columns:
        vel = df.drop_duplicates(subset=["REG_DATE_ONLY"])[["REG_DATE_ONLY", "REG_VELOCITY"]].copy()
        vel = vel.dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(vel["REG_VELOCITY"], bins=80, log=True)
        plt.title("Daily registration volume distribution (log y)")
        plt.xlabel("Registrations per day")
        plt.ylabel("Count (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "reg_velocity_dist.png"), dpi=150)
        plt.close()

    # 8 duplicate entity summary
    if "DUP_COUNT" in df.columns:
        dup_summary = df[df["DUP_COUNT"] > 1].groupby("DUP_COUNT")["SOS_VOTERID"].count().reset_index()
        dup_summary.columns = ["DUP_COUNT", "NUM_RECORDS"]
        dup_summary.to_csv(os.path.join(out_dir, "duplicate_entity_summary.csv"), index=False)
        print(f"[eda] dup entity summary saved, groups with dup>1: {len(dup_summary)}")

    print(f"[eda] all plots saved to {out_dir}")


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
    vel = df["REG_VELOCITY"].astype(float).values

    # rule based flags
    a1 = zscore_flag(age, thr=3.0)
    a2 = iqr_flag(age, k=1.5)
    d1 = zscore_flag(den, thr=3.0)
    d2 = iqr_flag(den, k=1.5)
    v1 = zscore_flag(vel, thr=3.0)
    dup_f = df["DUP_FLAG"].values

    rule_score = a1 + a2 + d1 + d2 + v1 + dup_f
    df["RULE_FLAG"] = (rule_score >= 1).astype(int)

    # isolation forest with 3 features
    X = np.vstack([age, den, vel]).T
    med_age = np.nanmedian(age)
    med_den = np.nanmedian(den)
    med_vel = np.nanmedian(vel)
    X[np.isnan(X[:, 0]), 0] = med_age
    X[np.isnan(X[:, 1]), 1] = med_den
    X[np.isnan(X[:, 2]), 2] = med_vel

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


def run_ablation(df, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)

    age = df["AGE"].astype(float).values
    den = df["ADDR_DENSITY"].astype(float).values
    vel = df["REG_VELOCITY"].astype(float).values
    dup_f = df["DUP_FLAG"].values

    # build feature matrix once
    X = np.vstack([age, den, vel]).T
    med_age = np.nanmedian(age)
    med_den = np.nanmedian(den)
    med_vel = np.nanmedian(vel)
    X[np.isnan(X[:, 0]), 0] = med_age
    X[np.isnan(X[:, 1]), 1] = med_den
    X[np.isnan(X[:, 2]), 2] = med_vel

    total = len(df)

    # --- part 1: rules-only vs IF-only vs combined ---
    a1 = zscore_flag(age, thr=3.0)
    a2 = iqr_flag(age, k=1.5)
    d1 = zscore_flag(den, thr=3.0)
    d2 = iqr_flag(den, k=1.5)
    v1 = zscore_flag(vel, thr=3.0)
    rule_score = a1+a2+d1+d2+v1+dup_f
    rules_only = (rule_score >= 1).astype(int).sum()

    m_base = IsolationForest(n_estimators=200, contamination=0.005, random_state=seed)
    m_base.fit(X)
    if_only = (m_base.predict(X) == -1).sum()

    combined = ((rule_score >= 1).astype(int) + (m_base.predict(X) == -1).astype(int))
    combined_cnt = (combined >= 1).sum()

    rows_1 = []
    rows_1.append({"method": "Rules-only", "anomalies": int(rules_only), "pct": round(float(rules_only)/total*100, 4)})
    rows_1.append({"method": "IF-only", "anomalies": int(if_only), "pct": round(float(if_only)/total*100, 4)})
    rows_1.append({"method": "Rules+IF", "anomalies": int(combined_cnt), "pct": round(float(combined_cnt)/total*100, 4)})
    pd.DataFrame(rows_1).to_csv(os.path.join(out_dir, "ablation_method_compare.csv"), index=False)
    print("[ablation] method comparison saved")

    # --- part 2: contamination sensitivity ---
    contam_vals = [0.001, 0.005, 0.01, 0.02, 0.05]
    rows_2 = []
    i = 0
    while i < len(contam_vals):
        c = contam_vals[i]
        m_c = IsolationForest(n_estimators=200, contamination=c, random_state=seed)
        m_c.fit(X)
        pred_c = m_c.predict(X)
        cnt_c = int((pred_c == -1).sum())
        rows_2.append({"contamination": c, "anomalies": cnt_c, "pct": round(float(cnt_c)/total*100, 4)})
        i += 1
    pd.DataFrame(rows_2).to_csv(os.path.join(out_dir, "ablation_contamination.csv"), index=False)
    print("[ablation] contamination sensitivity saved")

    # --- part 3: n_estimators sensitivity ---
    nest_vals = [50, 100, 200, 500]
    rows_3 = []
    i = 0
    while i < len(nest_vals):
        n = nest_vals[i]
        m_n = IsolationForest(n_estimators=n, contamination=0.005, random_state=seed)
        m_n.fit(X)
        pred_n = m_n.predict(X)
        cnt_n = int((pred_n == -1).sum())
        rows_3.append({"n_estimators": n, "anomalies": cnt_n, "pct": round(float(cnt_n)/total*100, 4)})
        i += 1
    pd.DataFrame(rows_3).to_csv(os.path.join(out_dir, "ablation_n_estimators.csv"), index=False)
    print("[ablation] n_estimators sensitivity saved")

    # --- plot contamination sensitivity ---
    c_df = pd.DataFrame(rows_2)
    plt.figure(figsize=(6, 4))
    plt.plot(c_df["contamination"], c_df["anomalies"], marker="o")
    plt.title("Anomalies vs contamination")
    plt.xlabel("contamination")
    plt.ylabel("anomalies flagged")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_contamination.png"), dpi=150)
    plt.close()

    # --- plot n_estimators sensitivity ---
    n_df = pd.DataFrame(rows_3)
    plt.figure(figsize=(6, 4))
    plt.plot(n_df["n_estimators"], n_df["anomalies"], marker="s")
    plt.title("Anomalies vs n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("anomalies flagged")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_n_estimators.png"), dpi=150)
    plt.close()

    print("[ablation] all done")


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
    p.add_argument("--ablation_dir", default="output/ablation")
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
    run_ablation(df, args.ablation_dir, args.seed)
    df.to_csv(args.all_out, index=False)
    save_top(df, args.out, args.top_n)


if __name__ == "__main__":
    main()
