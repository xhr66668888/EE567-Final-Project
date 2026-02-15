# EE567 Final Report

## 1. Project Title and Group Members
- **Title:** Auditing Ohio's Electoral Integrity: Unsupervised Anomaly Detection in Statewide Voter Registration Databases
- **Members:** Dingnan Huang, Haoran Xu

## 2. Roles and Responsibilities
- **Dingnan Huang:** Data acquisition, preprocessing pipeline, feature engineering, EDA visualization, ablation experiments, report writing.
- **Haoran Xu:** Isolation Forest modeling, rule-based detection system, combined scoring framework, pipeline automation, code integration, report writing.

## 3. Security Problem Definition

Voter roll integrity is a critical component of election cybersecurity. Statewide voter registration databases contain millions of records, and manual auditing at this scale is infeasible. Anomalies in these databases — such as voters with impossible ages, suspiciously high numbers of registrations at a single address, or abrupt registration spikes — may indicate data-quality failures, clerical errors, or deliberate manipulation.

We formulate this as an **unsupervised anomaly detection** problem because no ground-truth fraud labels exist. Our threat model considers three attack vectors:

1. **Identity anomalies:** Voters with biologically impossible ages (e.g., >110 years) still listed as ACTIVE, or duplicate entities sharing the exact same name and date of birth.
2. **Registration spikes:** Unnaturally large numbers of registrations on a single day, potentially indicating automated bulk-registration abuse.
3. **High-density addresses:** Implausibly many unrelated voters registered at a single residential address, suggesting either a data-entry error or a mailbox-farming scheme.

This project aligns with the course theme of applying machine learning to cybersecurity by treating public-record integrity auditing as a data-driven security problem.

## 4. Dataset Description

**Source:** Ohio Secretary of State, Statewide Voter File (SWVF) download portal (https://www6.ohiosos.gov/).

**Size:** Four gzip-compressed CSV files covering all 88 Ohio counties:
- `SWVF_1_22.txt.gz` (Counties 1–22)
- `SWVF_23_44.txt.gz` (Counties 23–44)
- `SWVF_45_66.txt.gz` (Counties 45–66)
- `SWVF_67_88.txt.gz` (Counties 67–88)

After merging: **7,899,468 records × 135 columns** (~550 MB compressed, ~1.7 GB uncompressed as CSV).

**Key attributes used:**

| Column | Description | Role |
|--------|-------------|------|
| `SOS_VOTERID` | Unique voter identifier | Record key |
| `DATE_OF_BIRTH` | Voter's date of birth | Age feature |
| `REGISTRATION_DATE` | Date of voter registration | Velocity feature |
| `RESIDENTIAL_ADDRESS1` | Street address line 1 | Density feature |
| `RESIDENTIAL_CITY` / `STATE` / `ZIP` | City, state, ZIP | Address normalization |
| `VOTER_STATUS` | ACTIVE / INACTIVE / CONFIRMATION | Filtering |
| `LAST_NAME` / `FIRST_NAME` | Voter name | Duplicate detection |
| `COUNTY_NUMBER` | County identifier | EDA segmentation |

**Data limitations:**
- No fraud labels — evaluation is necessarily qualitative and heuristic-based.
- Windows-1252 encoding with non-UTF-8 characters required special handling.
- Some fields contain legacy data from bulk database migrations (e.g., a mass registration event on July 17, 1989 affecting 89,483 records).

## 5. Methods and Implementation

### 5.1 Preprocessing

The preprocessing pipeline (`src/load_and_preprocess.py`) performs the following steps:

1. **Auto-format detection:** File magic bytes are checked to determine gzip vs. plain text; the header line is parsed to detect the delimiter (comma, pipe, or tab).
2. **Encoding handling:** Files are read with `latin-1` encoding to accommodate Windows-1252 characters.
3. **Date parsing:** `DATE_OF_BIRTH` and `REGISTRATION_DATE` are converted to pandas `Timestamp` objects using `pd.to_datetime()` with `errors="coerce"` for graceful handling of malformed dates.
4. **Address normalization:** Address fields are uppercased, whitespace-collapsed, and concatenated into a composite `ADDR_KEY` of the form `"<ADDR1> <ADDR2>, <CITY>, <STATE> <ZIP>"`.
5. **Column selection:** The dataset is reduced from 135 columns to 14 relevant fields, then saved as Parquet for efficient downstream processing.

### 5.2 Feature Engineering

Three numeric features are computed in `src/eda_and_baseline.py`:

- **Age:** $(T_{\text{now}} - T_{\text{DOB}}) / 365.25$, yielding fractional years. Missing DOBs produce `NaN`, imputed with the median.
- **Address density:** For each unique `ADDR_KEY`, we count distinct `SOS_VOTERID` values and merge the count back. This captures how many voters share the same physical address.
- **Registration velocity:** For each unique registration date, we count the number of registrations and merge back. This captures temporal clustering.

An auxiliary **duplicate-entity flag** is computed: records sharing the same `(LAST_NAME, FIRST_NAME, DATE_OF_BIRTH)` triple are flagged with `DUP_FLAG=1`.

### 5.3 Algorithms

#### 5.3.1 Statistical Baseline (Rules-Only)

We apply two complementary statistical tests on each of the three features:

- **Z-score flag:** A record is flagged if its standardized score exceeds ±3.0 standard deviations from the mean.
- **IQR flag:** A record is flagged if it falls below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$.

The rule score is the sum of all individual flags (age Z-score, age IQR, density Z-score, density IQR, velocity Z-score, duplicate flag). A record is considered rules-anomalous if the total score ≥ 1.

#### 5.3.2 Isolation Forest

We use scikit-learn's `IsolationForest` with the following configuration:
- **Features:** 3-dimensional input vector $(Age, AddressDensity, RegistrationVelocity)$
- **n_estimators:** 200 trees
- **contamination:** 0.005 (expected 0.5% anomaly rate)
- **random_state:** 567 (for reproducibility)

Missing values are imputed with column medians before fitting. The model outputs a continuous `decision_function` score (lower = more anomalous) and a binary prediction (`-1` = anomaly, `1` = normal).

#### 5.3.3 Combined Scoring

The final anomaly label is the union of rules-based and IF-based flags:

$$\text{FINAL\_ANOMALY} = \mathbb{1}[\text{RULE\_FLAG} + \text{IF\_FLAG} \geq 1]$$

This ensures high recall: any record flagged by either method is included in the anomaly set.

### 5.4 Ablation Study

We conducted three ablation experiments (`src/eda_and_baseline.py`, function `run_ablation`):

1. **Method comparison:** Rules-only vs. IF-only vs. Combined, measuring the number and percentage of flagged records.
2. **Contamination sensitivity:** Testing contamination values in {0.001, 0.005, 0.01, 0.02, 0.05} while holding other parameters fixed.
3. **n_estimators sensitivity:** Testing tree counts in {50, 100, 200, 500} while holding contamination at 0.005.

## 6. What We Implemented Ourselves

All code in this project was implemented by our team from scratch:

- **Data pipeline** (`src/load_and_preprocess.py`): Multi-file ingestion with auto-detection of compression, delimiter, and encoding; address normalization logic; Parquet export.
- **Feature engineering** (`src/eda_and_baseline.py`): Age computation, address density aggregation, registration velocity aggregation, duplicate-entity detection.
- **Detection logic** (`src/eda_and_baseline.py`): Z-score and IQR flagging functions, Isolation Forest wrapper, combined scoring, ablation framework.
- **EDA visualization** (`src/eda_and_baseline.py`): Six matplotlib plots covering age distribution, registration timeline, address density, county voter counts, age-by-status box plot, and registration velocity distribution.
- **Orchestration** (`run_pipeline.py`, `scripts/run_on_ece_lab.sh`): End-to-end pipeline runner with auto-detection of data file formats, automatic Git LFS handling for ECE lab deployment.

We used the following third-party libraries (not our own implementation): pandas, numpy, scikit-learn (`IsolationForest`), matplotlib, pyarrow.

## 7. Experimental Results and Analysis

### 7.1 Detection Summary

| Method | Anomalies Flagged | Percentage of Total |
|--------|------------------:|--------------------:|
| Rules-only | 644,655 | 8.16% |
| IF-only | 39,460 | 0.50% |
| Rules + IF (union) | 644,655 | 8.16% |

**Key finding:** All 39,460 IF-flagged records are a strict subset of the 644,655 rules-flagged records. This indicates that the Isolation Forest, at `contamination=0.005`, identifies the most extreme multivariate outliers — records that are already individually anomalous on at least one statistical dimension. The rules-based approach casts a much wider net due to the additive nature of its six binary flags.

### 7.2 Top Anomalous Records

The top-5 most anomalous records by IF decision score:

| Rank | Voter ID | Age | Addr Density | Reg Velocity | Address |
|------|----------|----:|------------:|--------------:|---------|
| 1 | OH0013908086 | 103.5 | 95 | 89,483 | 5900 Delhi Rd, Mt St Joseph |
| 2 | OH0013577660 | 99.8 | 95 | 89,483 | 5900 Delhi Rd, Mt St Joseph |
| 3 | OH0013605075 | 97.6 | 95 | 89,483 | 5900 Delhi Rd, Mt St Joseph |
| 4 | OH0013605074 | 97.0 | 95 | 89,483 | 5900 Delhi Rd, Mt St Joseph |
| 5 | OH0013860844 | 92.1 | 9 | 89,483 | 5156 North Bend Crossing, Cincinnati |

These top records combine three anomaly dimensions simultaneously: extreme age (>90 years), high address density, and the mass-registration-date spike. The address 5900 Delhi Rd corresponds to a known senior care facility (Mt. St. Joseph), which explains the co-occurrence of high age and high density — these are likely legitimate but warrant verification that all listed individuals are still alive and eligible.

### 7.3 High-Density Address Analysis

| Rank | Address | Voters |
|------|---------|-------:|
| 1 | 2100 Lakeside Ave, Cleveland, OH 44114 | 1,053 |
| 2 | 595 Van Buren Dr, Columbus, OH 43223 | 654 |
| 3 | 1400 Brush Row Rd, Wilberforce, OH 45384 | 435 |
| 4 | 924 E Main St, Columbus, OH 43205 | 361 |
| 5 | 2227 Payne Ave, Cleveland, OH 44114 | 323 |

The top address (2100 Lakeside Ave with 1,053 voters) likely corresponds to a large apartment complex or public housing facility. Nevertheless, such extreme density deserves county-level audit to confirm that records are not stale or duplicated.

### 7.4 Duplicate Entity Analysis

| Duplicate Count | Number of Records |
|---------------:|------------------:|
| 2 (pairs) | 10,874 |
| 3 (triples) | 90 |
| 4 (quadruples) | 12 |

5,437 pairs of records share the exact same (last name, first name, date of birth). While some are legitimate (e.g., a father and son named "John Smith" born on different dates that happen to be parsed the same), triples and quadruples are highly suspicious and should be flagged for manual review.

### 7.5 Ablation: Contamination Sensitivity

| Contamination | Anomalies | Percentage |
|--------------:|----------:|----------:|
| 0.001 | 7,898 | 0.10% |
| 0.005 | 39,460 | 0.50% |
| 0.01 | 78,841 | 1.00% |
| 0.02 | 157,905 | 2.00% |
| 0.05 | 394,949 | 5.00% |

The relationship between contamination and the number of flagged records is nearly perfectly linear, confirming that the Isolation Forest is well-calibrated: it flags approximately the expected fraction of records at each setting.

### 7.6 Ablation: n_estimators Sensitivity

| n_estimators | Anomalies | Percentage |
|-------------:|----------:|----------:|
| 50 | 39,497 | 0.50% |
| 100 | 39,302 | 0.50% |
| 200 | 39,460 | 0.50% |
| 500 | 39,492 | 0.50% |

Varying the number of trees from 50 to 500 has negligible impact on the anomaly count (~39,300–39,500 in all cases). This demonstrates that the model is **stable** and converges quickly; 200 trees is sufficient for reproducible results.

### 7.7 EDA Visualizations

The following plots are generated in `output/eda/`:

1. **`age_distribution.png`** — Histogram of voter ages, showing a primary mode around 40–60 years with a long right tail extending past 100.
2. **`registration_timeline.png`** — Time series of daily registration counts, revealing several historic spikes and a generally increasing trend in recent years.
3. **`addr_density_dist.png`** — Log-scale histogram of address density. The vast majority of addresses have 1–3 voters, but the tail extends to >1,000.
4. **`county_voter_count.png`** — Bar chart of voter counts per county, with the largest counties (Franklin, Cuyahoga, Hamilton) dominating.
5. **`age_boxplot_by_status.png`** — Box plots comparing age distributions across voter statuses (ACTIVE, INACTIVE, CONFIRMATION).
6. **`reg_velocity_dist.png`** — Log-scale histogram of daily registration volumes.

Additional ablation plots in `output/ablation/`:
- **`ablation_contamination.png`** — Line plot showing the linear relationship between contamination and anomaly count.
- **`ablation_n_estimators.png`** — Line plot confirming model stability across tree counts.

## 8. Future Work

1. **Improved deduplication:** Use fuzzy string matching (e.g., Levenshtein distance) on names combined with DOB proximity to catch typo-squatting duplicates that exact matching misses.
2. **Spatial analysis:** Geocode high-density addresses and cross-reference with public records (apartment buildings, nursing homes, shelters) to reduce false positives.
3. **Temporal decomposition:** Separate legitimate historical bulk-import events (e.g., the 1989 migration) from genuinely suspicious recent spikes using changepoint detection.
4. **Human-in-the-loop dashboard:** Build an interactive tool where auditors can review flagged records, annotate them, and iteratively refine the detection threshold.
5. **Cross-state comparison:** Apply the same pipeline to voter files from other states to test generalization.

## 9. Reproducibility

**Code repository:** https://github.com/xhr66668888/EE567-Final-Project

**Runtime environment:**
- Python 3.12+
- Dependencies: `pip install -r requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, requests, pyarrow)
- Tested on: UW ECE Linux Lab (AlmaLinux 8, linux-lab-129.ece.uw.edu)

**How to run:**
```bash
# Clone and run the full pipeline
git clone https://github.com/xhr66668888/EE567-Final-Project.git
cd EE567-Final-Project
bash scripts/run_on_ece_lab.sh
```

The script automatically:
1. Detects and installs `git-lfs` if needed (for pulling large data files)
2. Creates a Python virtual environment
3. Installs dependencies
4. Runs the full pipeline (preprocessing → EDA → Isolation Forest → ablation)

Outputs are saved to the `output/` directory.

## 10. References

1. Ohio Secretary of State. *Statewide Voter Files Download Page.* https://www6.ohiosos.gov/
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. In *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM)*, pp. 413–422.
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), Article 15.
4. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. In *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, pp. 93–104.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830.
6. Halder, S. & Ozdemir, S. (2018). *Hands-On Machine Learning for Cybersecurity.* Packt Publishing.
