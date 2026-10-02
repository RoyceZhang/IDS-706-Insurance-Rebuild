# Medical Insurance Cost Analysis Rebuild

This is **IDS 706 Repository B, Option 2: Rebuild My Previous Project**. It is a
beginner-friendly command-line project that validates medical insurance data,
summarizes important relationships, and demonstrates multiple linear regression.

The analysis reports selected statistics, average charges by smoking status and
region, and the Pearson association between BMI and charges. The model predicts
charges from `age`, `bmi`, `children`, and `smoker` (`no = 0`, `yes = 1`) using a
fixed 80/20 split and `random_state=42`. It reports MAE, MSE, RMSE, and R-squared.

## Project structure

```text
.
├── README.md
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── data/
│   ├── README.md
│   └── insurance.csv       # input dataset
├── src/
│   ├── __init__.py
│   ├── insurance_analysis.py
│   └── main.py
├── tests/
│   └── test_insurance_analysis.py
├── outputs/
│   └── .gitkeep            # generated results are ignored by Git
└── docs/
    ├── plan.md
    └── transcripts/
        └── .gitkeep
```

`src/insurance_analysis.py` contains reusable validation, analysis, feature, and
model functions. `src/main.py` is the thin command-line entry point. Tests use
small synthetic datasets so that they do not depend only on the full CSV.

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dataset is stored at `data/insurance.csv`. Its required schema and validation
assumptions are documented in `data/README.md`.

## Usage

Run with the default input and output locations:

```bash
python -m src.main
```

Or choose other locations:

```bash
python -m src.main --data path/to/insurance.csv --output-dir outputs
```

The command writes `analysis_summary.json`, `model_metrics.json`, and
`test_predictions.csv`. Generated files are ignored by Git and can be recreated.

## Current automated results

The Builder ran the workflow on October 1, 2026. The input contained 1,338 rows;
the program removed one exact duplicate and analyzed 1,337 rows. Smokers had
higher average charges than non-smokers ($32,050.23 compared with $8,440.66), and
BMI had a weak positive correlation with charges (0.1984). These are associations,
not evidence that either characteristic causes higher charges.

On the fixed test split, the model produced the following metrics:

- MAE: $4,198.59
- MSE: 35,914,551.48
- RMSE: $5,992.88
- R-squared: 0.8046

These automated results are reproducible evidence from the Builder stage. They do
not replace the owner's manual smoke test below.

## Testing and code quality

```bash
python -m pytest -q
python -m black --check src tests
python -m flake8 src tests
```

To apply Black formatting, run `python -m black src tests`.

## Docker

Build and run the default analysis command:

```bash
docker build -t insurance-analysis .
docker run --rm insurance-analysis
```

Run tests inside the same image:

```bash
docker run --rm insurance-analysis python -m pytest -q
```

## Interpretation and limitations

Group averages and correlation describe associations in observational data; they
do not establish causation. Linear regression is intentionally simple and may
miss nonlinear relationships. The fixed split and metrics demonstrate a
reproducible workflow, not a production-grade evaluation. Minor floating-point
differences may occur across platforms.

## Manual Smoke Test

I manually verified the project before the Tester stage.

I ran:

```bash
python -m pytest
black --check src tests
flake8 src tests
python -m src.main
docker build -t insurance-rebuild .
docker run --rm insurance-rebuild
```
All tests and code-quality checks passed. The command-line workflow ran
successfully and produced the expected analysis and regression results.
The Docker image also built and ran successfully.

## AI-Assisted Workflow Reflection

### Architect

The Architect helped define the project structure, data validation strategy,
model design, Docker approach, testing strategy, and verification plan.

I asked the Architect to simplify the original design because I wanted the
project to remain beginner-friendly. I rejected Docker Compose and a more
complicated preprocessing pipeline because this project only needs one Python
container and simple binary encoding.

I also asked the Architect to reduce the number of source files. The final
design uses one reusable analysis module and one small command-line entry point.

### Builder

The Builder implemented the project according to `docs/plan.md`, including
data validation, descriptive analysis, Linear Regression, tests, Docker, and
documentation.

I reviewed the generated implementation and asked the Builder to keep the
project simple and verify that the implementation matched the Architect plan.

### Tester

The Tester independently compared the implementation with `docs/plan.md`,
reviewed typical and edge-case behavior, checked the test suite, and verified
the setup and Docker instructions.

I reviewed the Tester findings, fixed important issues, and reran the relevant
checks.

### Accepted AI Recommendation

I accepted the recommendation to encode smoking status as a simple binary
feature (`no = 0`, `yes = 1`). This keeps the implementation understandable
while improving the model compared with Repository A.

### Changed or Rejected AI Recommendation

I rejected unnecessary Docker Compose and a more complicated preprocessing
pipeline because they would add complexity without improving this small
single-container project.

### Independent Verification

I independently ran the test suite, formatting check, linting check,
command-line workflow, Docker build, and Docker container before finalizing
the repository.


