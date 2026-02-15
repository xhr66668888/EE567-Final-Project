# EE567 Checkpoint Report

## 1. Project Title and Team
- **Title:** Auditing Ohio's Electoral Integrity: Unsupervised Anomaly Detection in Statewide Voter Registration Databases
- **Members:** Dingnan Huang, Haoran Xu

## 2. Milestones From Proposal
- [x] Data acquisition & merge 4 files
- [x] Preprocessing and address normalization
- [x] EDA (age + registration timeline)
- [x] Baseline Isolation Forest
- [x] Preliminary anomaly list

All five milestones established in the proposal have been completed.

## 3. What We Finished So Far

**Data pipeline.** We built an automated pipeline (`run_pipeline.py`) that ingests, merges, and preprocesses the four Ohio SWVF gzip files (counties 1–88). The merged dataset contains **7,899,468 voter records** across 135 raw columns. Preprocessing includes date parsing (`DATE_OF_BIRTH`, `REGISTRATION_DATE`), address normalization (uppercase, whitespace collapse, composite `ADDR_KEY`), and column selection down to the 14 fields relevant for anomaly detection.

**Feature engineering.** Three numeric features were engineered for modeling:
- **Age** — derived from `DATE_OF_BIRTH`, measured in fractional years.
- **Address density** — number of distinct `SOS_VOTERID` values sharing the same normalized address key.
- **Registration velocity** — number of voters registered on the same calendar date.

An additional duplicate-entity flag (`DUP_FLAG`) was computed by grouping on `(LAST_NAME, FIRST_NAME, DATE_OF_BIRTH)`.

**EDA.** Six exploratory visualizations were generated:
1. Age distribution histogram — reveals a heavy right tail with voters over age 100.
2. Registration timeline — shows clear spikes on certain dates (e.g., 89,483 registrations on 1989-07-17).
3. Address density distribution (log-scale) — most addresses have 1–2 voters; outliers exceed 1,000.
4. County-level voter count bar chart — identifies the most populated counties.
5. Age box plot by voter status — compares age distributions across ACTIVE, INACTIVE, CONFIRMATION, etc.
6. Daily registration volume distribution — highlights the long tail of registration spikes.

**Baseline model.** We implemented two detection approaches and a combined system:
- *Rules-only*: Z-score (threshold 3.0) and IQR (k=1.5) flags on age, address density, and registration velocity, plus the duplicate-entity flag. This flagged **644,655 records (8.16%)**.
- *Isolation Forest*: 200-tree ensemble with `contamination=0.005`, trained on (age, density, velocity). This flagged **39,460 records (0.50%)**.
- *Combined*: Union of rule-flagged and IF-flagged records. The combined set equals the rules-only set (644,655), meaning all IF-flagged records were already captured by rules.

**Top-500 anomaly list.** The 500 most anomalous records (sorted by ascending IF decision score) were exported. Top entries show ages above 100, address densities of 95+, and registration velocity of 89,483 — all concentrated at 5900 Delhi Rd, Mt St Joseph (a known nursing-home facility with 95 registered voters).

## 4. Preliminary Results

| Method | Anomalies Flagged | Percentage |
|--------|------------------:|----------:|
| Rules-only | 644,655 | 8.16% |
| IF-only | 39,460 | 0.50% |
| Rules + IF (union) | 644,655 | 8.16% |

**Top suspicious patterns:**
- **Impossible ages:** Multiple active voters with computed age > 100 years (oldest: 103.5 years, DOB 1922-08-12, still ACTIVE status).
- **High-density addresses:** 2100 Lakeside Ave, Cleveland has 1,053 registered voters at a single address; 595 Van Buren Dr, Columbus has 654.
- **Registration spike:** July 17, 1989 had 89,483 registrations in a single day — likely a historical bulk-import event rather than fraud, but warrants verification.
- **Duplicate entities:** 5,437 pairs share the same (last name, first name, DOB); 45 triples and 3 quadruples exist.

**Initial conclusion:** The Isolation Forest effectively isolates the most extreme multivariate outliers (high age + high density + high velocity). The rules-based approach casts a wider net. The union strategy ensures no IF-detected anomaly is missed while preserving rule-based coverage.

## 5. Bottlenecks and Mitigation

- **Bottleneck 1:** Large file size (~550 MB compressed, ~8M rows) caused memory pressure on lab machines during the mixed-evaluation step.
  - *Mitigation:* We implemented stratified sampling (50K real records + 200 synthetic) for the adversarial evaluation, keeping the core detection on the full dataset.

- **Bottleneck 2:** SWVF files use Windows-1252 encoding with non-UTF-8 bytes (e.g., 0x92 = right single quotation mark), causing `UnicodeDecodeError`.
  - *Mitigation:* Switched to `latin-1` encoding in `pd.read_csv()` with `on_bad_lines="warn"`.

## 6. Next Experiments Before Final

- ~~Hyperparameter sensitivity on contamination~~ ✅ Done (ablation study completed)
- ~~Compare IF vs. rules-only baseline~~ ✅ Done
- Qualitative case-study analysis of top-10 flagged records
- Manual verification of high-density addresses (nursing homes vs. warehouses)

## 7. Open Questions

- **Q1:** High-density addresses like 2100 Lakeside Ave (1,053 voters) — is this a legitimate large residential facility or a data-entry anomaly? County-level cross-referencing would help.
- **Q2:** The July 1989 registration spike affects 89,483 records. Is this a statewide database migration event? Historical SOS records could clarify.

## 8. References
1. Ohio Secretary of State. Statewide Voter Files Download Page. https://www6.ohiosos.gov/
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM 2008*.
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*.
4. Breunig, M. M., et al. (2000). LOF: Identifying Density-Based Local Outliers. *SIGMOD 2000*.
