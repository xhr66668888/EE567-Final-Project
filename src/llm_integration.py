import argparse
import json
import os
import random
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MINIMAX_API_KEY = "sk-api-l1L7IaUbRR6EiGJ6SJ-FESmovymV9n5R9M6I0G-i02DFH6CYm64X8fRCIgpR2IU61q1vkIqZyFA4MLbZ2Em7CxPsycoriBCB1g7tiOHFx0bID_4gGpq_vyE"
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = "MiniMax-Text-01"


def read_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _post_chat(messages, temperature=0.2):
    key = MINIMAX_API_KEY
    base = MINIMAX_BASE_URL
    model = MINIMAX_MODEL

    if key == "":
        return None, "no api key"

    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            return None, f"http {r.status_code}: {r.text[:300]}"
        data = r.json()
        txt = data["choices"][0]["message"]["content"]
        return txt, "ok"
    except Exception as e:
        return None, str(e)


def _rule_based_fake(df, n):
    out = []
    # sample a small pool instead of converting millions of rows to list
    sample_size = min(5000, len(df))
    sampled = df.sample(n=sample_size, random_state=42)
    pool_addr = sampled["ADDR_KEY"].dropna().astype(str).tolist()
    pool_city = sampled["RESIDENTIAL_CITY"].dropna().astype(str).tolist()

    i = 0
    while i < n:
        addr = ""
        city = "COLUMBUS"
        if len(pool_addr) > 0:
            addr = pool_addr[random.randint(0, len(pool_addr)-1)]
        if len(pool_city) > 0:
            city = pool_city[random.randint(0, len(pool_city)-1)]

        row = {
            "SOS_VOTERID": f"SYNTH_{i}",
            "COUNTY_NUMBER": "99",
            "COUNTY_ID": f"SYNTH_{100000+i}",
            "LAST_NAME": f"SM1TH{i}",
            "FIRST_NAME": "J0HN",
            "DATE_OF_BIRTH": "1890-01-01",
            "REGISTRATION_DATE": "2025-10-31",
            "VOTER_STATUS": "ACTIVE",
            "RESIDENTIAL_ADDRESS1": "1 TEST WAREHOUSE DR",
            "RESIDENTIAL_SECONDARY_ADDR": "UNIT 999",
            "RESIDENTIAL_CITY": city,
            "RESIDENTIAL_STATE": "OH",
            "RESIDENTIAL_ZIP": "43215",
            "ADDR_KEY": "1 TEST WAREHOUSE DR UNIT 999, COLUMBUS, OH 43215",
            "IS_SYNTH": 1
        }
        if i % 3 == 0 and addr != "":
            row["ADDR_KEY"] = addr
        out.append(row)
        i += 1
    return pd.DataFrame(out)


def generate_synthetic_with_llm(df, n=200):
    sample_cols = [
        "SOS_VOTERID", "COUNTY_NUMBER", "COUNTY_ID", "LAST_NAME", "FIRST_NAME",
        "DATE_OF_BIRTH", "REGISTRATION_DATE", "VOTER_STATUS",
        "RESIDENTIAL_ADDRESS1", "RESIDENTIAL_SECONDARY_ADDR", "RESIDENTIAL_CITY",
        "RESIDENTIAL_STATE", "RESIDENTIAL_ZIP", "ADDR_KEY"
    ]

    base_rows = []
    tmp = df[sample_cols].head(8).fillna("")
    i = 0
    while i < len(tmp):
        row = tmp.iloc[i].to_dict()
        # convert non-serializable types (Timestamp etc.) to strings
        for k, v in row.items():
            if not isinstance(v, (str, int, float, bool, type(None))):
                row[k] = str(v)
        base_rows.append(row)
        i += 1

    sys_msg = {
        "role": "system",
        "content": "You are a cybersecurity data generator. Return strict JSON only."
    }
    usr_msg = {
        "role": "user",
        "content": (
            "Generate " + str(n) + " synthetic fraudulent Ohio voter records for adversarial testing. "
            "Attack vectors: impossible ages, typo-squatting names, unrealistic high-density shared addresses, unnatural registration spikes. "
            "Keep same keys as examples. Return JSON array only. Examples: " + json.dumps(base_rows)
        )
    }

    txt, msg = _post_chat([sys_msg, usr_msg], temperature=0.8)
    if txt is None:
        print(f"[llm] fallback because {msg}")
        return _rule_based_fake(df, n)

    try:
        t = txt.strip()
        if t.startswith("```"):
            t = t.replace("```json", "").replace("```", "").strip()
        arr = json.loads(t)
        fake = pd.DataFrame(arr)
        fake["IS_SYNTH"] = 1
        return fake
    except Exception as e:
        print(f"[llm] parse fail, fallback {e}")
        return _rule_based_fake(df, n)


def add_features(df):
    now = pd.Timestamp.now()
    dob = pd.to_datetime(df["DATE_OF_BIRTH"], errors="coerce")
    reg = pd.to_datetime(df["REGISTRATION_DATE"], errors="coerce")
    age_days = (now - dob).dt.days
    df["AGE"] = age_days / 365.25

    den = df.groupby("ADDR_KEY")["SOS_VOTERID"].nunique().reset_index()
    den.columns = ["ADDR_KEY", "ADDR_DENSITY"]
    df = df.merge(den, on="ADDR_KEY", how="left")
    df["REG_DATE_ONLY"] = reg.dt.date
    return df


def run_if(df, contamination=0.005, seed=567):
    age = df["AGE"].astype(float).values
    den = df["ADDR_DENSITY"].astype(float).values
    X = np.vstack([age, den]).T

    med_age = np.nanmedian(age)
    med_den = np.nanmedian(den)
    X[np.isnan(X[:, 0]), 0] = med_age
    X[np.isnan(X[:, 1]), 1] = med_den

    m = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    m.fit(X)
    pred = m.predict(X)
    score = m.decision_function(X)
    df["IF_FLAG"] = (pred == -1).astype(int)
    df["IF_SCORE"] = score
    return df


def eval_detect_rate(df_mix):
    t = df_mix[df_mix["IS_SYNTH"] == 1]
    if len(t) == 0:
        return 0.0
    hit = t["IF_FLAG"].sum()
    rate = float(hit) / float(len(t))
    return rate


def explain_one_row(row):
    payload = {
        "SOS_VOTERID": str(row.get("SOS_VOTERID", "")),
        "AGE": str(row.get("AGE", "")),
        "ADDR_DENSITY": str(row.get("ADDR_DENSITY", "")),
        "REGISTRATION_DATE": str(row.get("REGISTRATION_DATE", "")),
        "IF_SCORE": str(row.get("IF_SCORE", "")),
        "VOTER_STATUS": str(row.get("VOTER_STATUS", "")),
        "ADDR_KEY": str(row.get("ADDR_KEY", ""))
    }

    sys_msg = {
        "role": "system",
        "content": "You are a Security Auditor. Be short and concrete."
    }
    usr_msg = {
        "role": "user",
        "content": "Explain why this voter record was flagged as anomaly in 2-3 sentences, include likely risk reason and one verification suggestion. Data: " + json.dumps(payload)
    }

    txt, msg = _post_chat([sys_msg, usr_msg], temperature=0.2)
    if txt is None:
        return f"Flagged by low IF score={payload['IF_SCORE']}, possible abnormal age/address density. Suggest manual county-level verification. ({msg})"
    return txt.strip()


def explain_top(df, n=20):
    out = []
    top = df.sort_values("IF_SCORE", ascending=True).head(n).copy()
    i = 0
    while i < len(top):
        row = top.iloc[i]
        exp = explain_one_row(row)
        item = {
            "SOS_VOTERID": row.get("SOS_VOTERID", ""),
            "IF_SCORE": row.get("IF_SCORE", ""),
            "AGE": row.get("AGE", ""),
            "ADDR_DENSITY": row.get("ADDR_DENSITY", ""),
            "EXPLANATION": exp
        }
        out.append(item)
        print(f"[llm-explain] {i+1}/{len(top)}")
        i += 1
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="output/swvf_processed.parquet")
    p.add_argument("--synthetic_n", type=int, default=200)
    p.add_argument("--explain_n", type=int, default=20)
    p.add_argument("--mix_out", default="output/mixed_with_synth.csv")
    p.add_argument("--eval_out", default="output/synth_eval.txt")
    p.add_argument("--explain_out", default="output/llm_explanations.csv")
    args = p.parse_args()

    real_df = read_df(args.input)
    real_df["IS_SYNTH"] = 0

    fake_df = generate_synthetic_with_llm(real_df, n=args.synthetic_n)

    cols = list(real_df.columns)
    i = 0
    while i < len(cols):
        c = cols[i]
        if c not in fake_df.columns:
            fake_df[c] = ""
        i += 1

    # use a stratified sample of real data to keep memory/time manageable
    sample_n = min(50000, len(real_df))
    real_sample = real_df.sample(n=sample_n, random_state=567)
    print(f"[mix] using {sample_n} real + {len(fake_df)} synth for evaluation")

    fake_df = fake_df[real_sample.columns]
    mix = pd.concat([real_sample, fake_df], axis=0, ignore_index=True)

    mix = add_features(mix)
    mix = run_if(mix)
    mix.to_csv(args.mix_out, index=False)

    rate = eval_detect_rate(mix)
    s = f"synthetic_detect_rate={rate:.4f}\nrows_total={len(mix)}\nrows_synth={int(mix['IS_SYNTH'].sum())}\n"
    with open(args.eval_out, "w", encoding="utf-8") as f:
        f.write(s)
    print(s)

    top = mix[mix["IF_FLAG"] == 1].copy()
    exp_df = explain_top(top, n=args.explain_n)
    exp_df.to_csv(args.explain_out, index=False)
    print(f"[save] {args.explain_out}")


if __name__ == "__main__":
    main()
