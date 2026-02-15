import argparse
import os
import subprocess


def run_cmd(cmd):
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError("command failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_llm", action="store_true")
    p.add_argument("--out_dir", default="output")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # support both .txt.gz and .txt file names
    gz_files = [
        "SWVF_1_22.txt.gz",
        "SWVF_23_44.txt.gz",
        "SWVF_45_66.txt.gz",
        "SWVF_67_88.txt.gz"
    ]
    txt_files = [
        "SWVF_1_22.txt",
        "SWVF_23_44.txt",
        "SWVF_45_66.txt",
        "SWVF_67_88.txt"
    ]
    files = []
    for gz, txt in zip(gz_files, txt_files):
        if os.path.isfile(gz):
            files.append(gz)
        elif os.path.isfile(txt):
            files.append(txt)
        else:
            print(f"[warn] neither {gz} nor {txt} found, skipping")
    if not files:
        raise RuntimeError("No SWVF data files found in repo root. "
                           "Place SWVF_*.txt or SWVF_*.txt.gz here.")

    p1 = [
        "python", "src/load_and_preprocess.py",
        "--files"
    ] + files + [
        "--out", f"{args.out_dir}/swvf_processed.parquet"
    ]
    run_cmd(p1)

    p2 = [
        "python", "src/eda_and_baseline.py",
        "--input", f"{args.out_dir}/swvf_processed.parquet",
        "--eda_dir", f"{args.out_dir}/eda",
        "--out", f"{args.out_dir}/anomalies_top500.csv",
        "--all_out", f"{args.out_dir}/scored_all.csv",
        "--top_n", "500",
        "--contamination", "0.005",
        "--seed", "567"
    ]
    run_cmd(p2)

    if not args.skip_llm:
        p3 = [
            "python", "src/llm_integration.py",
            "--input", f"{args.out_dir}/swvf_processed.parquet",
            "--synthetic_n", "200",
            "--explain_n", "20",
            "--mix_out", f"{args.out_dir}/mixed_with_synth.csv",
            "--eval_out", f"{args.out_dir}/synth_eval.txt",
            "--explain_out", f"{args.out_dir}/llm_explanations.csv"
        ]
        run_cmd(p3)

    print("[done] pipeline finished")


if __name__ == "__main__":
    main()
