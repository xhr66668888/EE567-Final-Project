# Project Proposal (EE567)

## Project Title
Auditing Ohio’s Electoral Integrity: Unsupervised Anomaly Detection in Statewide Voter Registration Databases

## Group Members
- Dingnan Huang
- Haoran Xu

## Dataset(s)
We use Ohio Statewide Voter Files (SWVF) from the Ohio Secretary of State portal: https://www6.ohiosos.gov/.

- Coverage: all 88 counties in Ohio
- Scale: around 8 million records
- Structure: 100+ attributes per voter, including demographics, date fields, and address fields
- Type: high-dimensional unlabeled tabular data

## Problem(s) to Work On
Voter roll integrity is part of election cybersecurity. Manual audit over millions of records is not feasible. We formulate this as unsupervised anomaly detection due to missing fraud labels.

Target anomalies:
- Identity anomalies: impossible ages, suspicious duplicate entities
- Registration spikes: abrupt volume bursts indicating potential automated registration abuse
- High-density addresses: implausibly many unrelated voters at one residential address

## Potential Ways to Solve the Problem(s)
### Feature Engineering
- Age from `DATE_OF_BIRTH`
- Address density from unique `SOS_VOTERID` per normalized residential address
- Temporal velocity from registrations per day/week

### Modeling
- Statistical baseline: Z-score and IQR flags on age and density
- Isolation Forest: unsupervised tree ensemble for anomaly isolation
- Autoencoder (optional extension): reconstruction error based anomaly scoring

### Evaluation Strategy (No Ground Truth Labels)
- Top-N qualitative inspection of suspicious records
- Heuristic cross-checks (e.g., nursing homes vs suspicious warehouse addresses)
- County/city-level sanity checks for false-positive reduction

## Milestone Deliverables
- Data ingestion: merge `SWVF_1_22` to `SWVF_67_88`
- Preprocessing: missing values, date parsing, address normalization
- EDA: age distribution and registration timeline baseline
- Baseline model: Isolation Forest and preliminary anomaly list

## Added Extension: Generative AI for Adversarial Testing & Interpretation
To reduce the unlabeled-data challenge, we add LLM integration in two ways:

1. Synthetic adversarial data generation
- Use an LLM to create realistic synthetic fraudulent records (e.g., typo-squatting names, impossible ages, extreme address density).
- Inject synthetic records into real data and evaluate whether Isolation Forest detects them.

2. Automated anomaly explanation
- For top suspicious records, send feature vectors to an LLM agent in “Security Auditor” role.
- Generate short natural-language justifications and practical verification suggestions.

## References
- Ohio Secretary of State. Statewide Voter Files Download Page.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest.
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey.
- Breunig, M. M., et al. (2000). LOF: Identifying Density-Based Local Outliers.
