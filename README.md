# EE567 Final Project

Auditing Ohio’s Electoral Integrity: Unsupervised Anomaly Detection in Statewide Voter Registration Databases

## 1) Project Goal (Course-Aligned)
This project follows the EE567 course project requirement for applying ML to cybersecurity-style data integrity problems.

We detect suspicious patterns in Ohio Statewide Voter Files (SWVF) with unlabeled anomaly detection:
- impossible voter age
- unnatural registration spikes
- very high-density shared addresses

We use methods covered by class level and not over-advanced:
- statistical baseline (Z-score + IQR)
- Isolation Forest (main baseline model)
- optional deep model extension (autoencoder, not required to run now)

## 2) Repository Structure
- `src/load_and_preprocess.py` merge 4 SWVF gzip files, clean basic columns, normalize address
- `src/eda_and_baseline.py` feature engineering + EDA + Isolation Forest + anomaly output
- `src/llm_integration.py` LLM synthetic adversarial record generation + anomaly explanation
- `run_pipeline.py` run all stages in order
- `reports/proposal_final.md` proposal draft
- `reports/checkpoint_report_template.md` checkpoint template (2 pages)
- `reports/final_report_template.md` final report template

## 3) Environment
Install packages:

```bash
pip install -r requirements.txt
```

## 4) Data Files Required
Put these files at repo root (already present in this workspace):
- `SWVF_1_22.txt.gz`
- `SWVF_23_44.txt.gz`
- `SWVF_45_66.txt.gz`
- `SWVF_67_88.txt.gz`

## 5) Run
Run full pipeline:

```bash
python run_pipeline.py
```

If you want to skip LLM stage:

```bash
python run_pipeline.py --skip_llm
```

## 6) Output Files
Main outputs under `output/`:
- `swvf_processed.parquet` processed merged data
- `eda/age_distribution.png` age histogram
- `eda/registration_timeline.png` registration timeline
- `scored_all.csv` all records with anomaly scores
- `anomalies_top500.csv` top suspicious records
- `mixed_with_synth.csv` dataset mixed with synthetic adversarial samples
- `synth_eval.txt` synthetic detection rate
- `llm_explanations.csv` natural-language explanations for top anomalies

## 7) LLM Integration (MiniMax)
The MiniMax API key is hardcoded in `src/llm_integration.py` for convenience.
No extra environment setup needed.

Run LLM stage:

```bash
python src/llm_integration.py --input output/swvf_processed.parquet
```

If API call fails for any reason, code automatically falls back to rule-based synthetic generation and template explanation.

## 8) How This Matches Grading Requirements
- **Project title / members / dataset / problem / methods / milestone / references**: in `reports/proposal_final.md`
- **Checkpoint report**: use `reports/checkpoint_report_template.md`, explicitly map done vs not-done milestones
- **Final report** (<=10 pages): use `reports/final_report_template.md`, includes roles/responsibilities, own implementation part, results/analysis, future work, references, code link
- **Code deliverable**: this repo + README run steps

## 9) Suggested Next Steps for Full Score
- Tune contamination and compare with rules-only baseline
- Add simple dedup heuristic (same name + DOB + address similarity)
- Do top-N manual audit notes for suspicious cases
- Put key figures/tables into slide deck and final report

## 10) Run on UW ECE Linux Lab (Remote)
This repo can run on AlmaLinux 8 servers (linux-lab-101 ~ linux-lab-148) by SSH.

### Step A: Connect
1. First connect Husky OnNet VPN.
2. Then SSH:

```bash
ssh yournetid@linux-lab-129.ece.uw.edu
```

### Step B: Clone and run

```bash
git clone https://github.com/xhr66668888/EE567-Final-Project.git
cd EE567-Final-Project
bash scripts/run_on_ece_lab.sh
```

### Step C: If NFS quota is tight, run in /var/tmp
Linux lab has shared NFS home quotas. For heavy output, run under local disk:

```bash
cp -r ~/EE567-Final-Project /var/tmp/$USER/EE567-Final-Project
cd /var/tmp/$USER/EE567-Final-Project
bash scripts/run_on_ece_lab.sh
```

### Step D: Check outputs
Main result files are in `output/`.

### Notes
- Script creates `.venv` automatically.
- Script installs dependencies from `requirements.txt`.
- SWVF data files are stored via **Git LFS**. The script auto-detects LFS pointers and installs `git-lfs` locally if needed.
- If MiniMax API call fails, pipeline still runs with fallback synthetic generation.
