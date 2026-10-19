# Week 2: Exploratory Data Analysis (EDA) and Visualization Framework

A reusable Python framework for profiling a CSV dataset, checking data quality, exploring distributions and relationships, generating plots, and writing a Markdown report.

> The assignment does not provide a dataset. A small synthetic sample dataset is included so the project can be tested immediately. Replace it with your own CSV to analyze a real dataset.

## Features
- Dataset loading and structural profiling
- Missing-value, duplicate-row, and duplicate-column checks
- Numeric and categorical summaries
- IQR-based outlier screening (flags only; does not automatically delete records)
- Univariate, bivariate, and correlation visualizations
- Automated Markdown report and CSV summaries
- Configurable input path and output directory

## Project structure
```text
Week-2-EDA-Visualization-Framework/
├── data/
│   └── sample_data.csv
├── figures/                  # generated plots
├── reports/                  # generated reports and summary tables
├── src/
│   ├── __init__.py
│   ├── eda.py
│   └── report.py
├── tests/
│   └── test_eda.py
├── .gitignore
├── requirements.txt
├── run_eda.py
└── README.md
```

## 1. Install Python
Use Python 3.10 or newer if possible.

## 2. Create and activate a virtual environment
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Run on the included sample data
```bash
python run_eda.py
```

## 5. Run on your own CSV
```bash
python run_eda.py --input data/your_dataset.csv --output reports/my_eda
```

The output directory will contain:
- `eda_report.md`: readable summary of the analysis
- `data_profile.csv`: data type, missingness, and unique-value profile
- `numeric_summary.csv`: descriptive statistics for numeric columns
- `categorical_summary.csv`: frequency summaries for categorical columns
- `outlier_flags.csv`: row indices and columns flagged by the IQR screening rule
- `figures/`: PNG visualizations

## Example output
The framework can produce:
- Numeric histograms and box plots
- Categorical frequency charts
- Scatter plots for selected numeric pairs
- A numeric correlation heat map
- A report summarizing dataset shape, missingness, duplicates, and possible outliers

## Important analytical notes
- Outliers are flagged for review, not automatically removed.
- Missing values are reported; the framework does not silently impute them.
- Correlation does not prove causation.
- The first two numeric columns are used for a simple scatter plot unless you configure your own plotting logic.
- Use the report as an exploratory starting point. Confirm any finding with domain knowledge before making decisions.

## Libraries
- `pandas`: tabular data processing and summaries
- `numpy`: numerical operations
- `matplotlib`: plotting
- `seaborn`: statistical visualization

## GitHub upload
From the project directory:
```bash
git init
git add .
git commit -m "Add EDA and visualization framework"
git branch -M main
git remote add origin https://github.com/Bhavikpatel2910/Week-2-EDA-Visualization-Framework.git
git push -u origin main
```
If the repository already has a remote or commits, do not run `git remote add origin` again. Check with `git remote -v` and `git status` first. Authenticate using GitHub's supported sign-in or token workflow; do not put credentials in source files.
