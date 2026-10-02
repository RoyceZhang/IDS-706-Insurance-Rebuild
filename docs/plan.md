# Architect Plan: Medical Insurance Cost Analysis Rebuild

## 1. Scope and success criteria

This repository will be a fresh, beginner-friendly rebuild for **IDS 706 Repository B, Option 2: Rebuild My Previous Project**. It will not copy the earlier project's code. It will use the Medical Cost Personal Dataset to:

- validate and clean the data;
- summarize relationships between insurance charges and age, BMI, smoking status, and region;
- train a multiple linear regression model using `age`, `bmi`, `children`, and an encoded `smoker` feature;
- produce deterministic, human-readable and machine-readable results; and
- support local and Docker-based formatting, linting, testing, and execution.

The project is complete when a clean checkout can install its pinned dependencies, pass Black, flake8, and pytest checks, run one analysis/model command, create the documented outputs, and do the same in a Python 3.11 container.

## 2. Key technical decisions

### Data contract and validation

`data/insurance.csv` is the single input. The loader will require these columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`. Extra columns may be retained by loading, but only documented columns will be used.

Processing order:

1. Read the CSV and fail with a clear error if it is unreadable or empty.
2. Check all required column names.
3. Remove exact duplicate rows and report the number removed.
4. Normalize surrounding whitespace and lowercase categorical values.
5. Reject missing values in required fields rather than silently imputing them.
6. Validate numeric fields: nonnegative age, children, and charges; positive BMI; integer-valued children.
7. Validate `smoker` as `yes` or `no`, and require nonblank `sex` and `region` values.

Assumptions and the dataset's exact source should be recorded in `data/README.md`. The original CSV must not be silently changed in place; cleaning happens in memory. This dataset is expected to be clean enough that rejecting invalid or missing required values is clearer than inventing imputation rules.

### Analysis

Keep the analysis descriptive and small. Produce:

- row count before and after duplicate removal;
- selected numeric summary statistics (count, mean, median, standard deviation, minimum, and maximum) for age, BMI, and charges;
- counts and mean charges grouped by smoker status;
- counts and mean charges grouped by region; and
- the Pearson correlation between BMI and charges, described as association rather than causation.

A dataset containing only smokers or only non-smokers is valid for descriptive analysis: return the observed group without failing. Avoid plots and statistical significance testing unless the owner explicitly expands the scope.

### Model

Use scikit-learn `LinearRegression` with exactly four features: `age`, `bmi`, `children`, and `smoker_encoded`. Encode smoker with the explicit mapping `no -> 0`, `yes -> 1`. This is simpler and more interpretable than a one-hot encoder for a validated binary field, and it avoids a redundant dummy column.

Do not use scikit-learn `Pipeline`, `ColumnTransformer`, scaling, imputation, polynomial features, or automated feature engineering. The dataset needs only direct validation plus the one explicit smoker mapping. These tools would be reasonable for a larger mixed-type project, but they add unnecessary concepts here.

Use an 80/20 train/test split with `random_state=42`. Report mean absolute error (MAE), root mean squared error (RMSE), and R-squared (R2), plus train/test row counts. Negative R2 is valid and must not be treated as a program error. Reject training data with fewer than 10 cleaned rows with a clear message; this is a guard against splits too small to yield meaningful test metrics, not a claim that 10 rows are statistically sufficient.

The main script should write deterministic outputs to `outputs/`:

- `analysis_summary.json` for descriptive results;
- `model_metrics.json` for metrics and split metadata; and
- `test_predictions.csv` for actual and predicted charges for the held-out rows.

Sort prediction output by original row index so repeated runs are easy to compare. JSON output must contain standard JSON values (use `null`, not nonstandard `NaN`, when a value cannot be computed).

### Application boundary

Use only two substantive source files. `src/insurance_analysis.py` will contain small, clearly named reusable functions for loading/validation, summaries, feature preparation, and model training. Keeping these related functions together is easier for a beginner to navigate than splitting a small project across three modules. Use section comments and docstrings to keep the file readable.

`src/main.py` will be the thin command-line entry point. It should accept `--data` and `--output-dir` arguments with defaults of `data/insurance.csv` and `outputs`, call functions from `insurance_analysis.py`, print a concise summary, and return a nonzero exit code with a useful message on invalid input. Calculations should remain in importable functions rather than being embedded in command-line parsing or printing code.

## 3. Risks and design concerns

- **Data provenance:** the Builder must use the supplied dataset and document its exact source; it must not guess a download URL or claim a license without evidence.
- **Data leakage:** deduplication and validation occur before splitting, and the target (`charges`) must never be used as an input feature. No learned preprocessing is planned.
- **Overclaiming:** this observational dataset supports prediction and association, not causal claims about smoking or BMI.
- **Model limitations:** linear regression is intentionally simple and may underfit nonlinear effects. The README should state this rather than adding complexity outside the assignment scope.
- **Metric instability:** R2 is unreliable on very small test sets, so training rejects fewer than 10 rows and the README describes metrics as a demonstration, not a production evaluation.
- **Platform consistency:** line endings, output ordering, fixed random state, and pinned direct dependencies should make local and Docker results reproducible. Minor floating-point differences across platforms may still occur.
- **Stale outputs:** generated artifacts can become inconsistent with code. The default recommendation is to ignore generated files and recreate them, then record verified key results in the README.
- **Scope creep:** no notebook, web service, Docker Compose, exposed port, database, advanced preprocessing, model search, or visualization dependency is needed. Docker Compose solves multi-container coordination, while this project has only one command-line container.

## 4. Final project structure

```text
.
├── README.md
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── data/
│   ├── README.md
│   └── insurance.csv
├── src/
│   ├── __init__.py
│   ├── insurance_analysis.py
│   └── main.py
├── tests/
│   └── test_insurance_analysis.py
├── outputs/
│   └── .gitkeep
└── docs/
    ├── plan.md
    └── transcripts/
        └── .gitkeep
```

Changes from the initial suggestion are justified as follows: all reusable logic is consolidated in `src/insurance_analysis.py`, while `src/main.py` remains a small runnable wrapper. One test file mirrors that small codebase. `data/README.md` records provenance and schema, and `outputs/` gives generated results a predictable location. A packaging file is unnecessary because commands run from the repository root with `python -m src.main`.

## 5. Builder implementation sequence

The Builder should implement only after receiving this plan in a new AI conversation.

1. Create the directory skeleton, ignore rules, dataset documentation, and a single `requirements.txt` with tested, exact versions of direct runtime and development dependencies: pandas, scikit-learn, pytest, Black, and flake8.
2. Add the provided `insurance.csv` unchanged and document its provenance, schema, and validation assumptions.
3. Implement small functions for loading, duplicate removal, normalization, validation, summaries, feature preparation, deterministic model training, prediction, and metrics in `src/insurance_analysis.py`. Keep feature names and the smoker mapping explicit, and keep calculation functions independent of console output.
4. Implement `src/main.py` to call those functions in order and serialize outputs. Create the output directory when needed.
5. Add focused tests in one test file using small synthetic DataFrames or temporary CSV files. Tests should not rely only on the full production dataset.
6. Add one minimal Dockerfile based on `python:3.11-slim`: set a work directory, copy and install requirements with `--no-cache-dir`, copy the repository, and run `python -m src.main` by default. Do not expose ports or add Docker Compose.
7. Write the README with purpose, selected option, structure, setup, usage, test/format/lint commands, Docker commands, limitations, and AI-workflow sections. Leave clearly marked places for actual manual smoke-test and Tester findings; do not fabricate results.
8. Run all quality gates, correct failures, generate outputs, and report exactly what was run. Do not change requirements merely to suppress a failing test without explaining the root cause.

## 6. Tests and acceptance checks

### Focused pytest coverage

`tests/test_insurance_analysis.py` should cover:

- a valid temporary CSV loads with the required columns and expected row count;
- exact duplicate rows are removed;
- a missing required column raises a clear error naming the column; and
- one parameterized validation test rejects representative missing, invalid categorical, and out-of-range numeric values;
- smoker and region summaries calculate correct counts and means on a tiny known dataset;
- a dataset with no smokers returns the non-smoker result without an exception; and
- the BMI/charges correlation matches a simple known relationship;
- feature preparation returns columns in the exact order `age`, `bmi`, `children`, `smoker_encoded` and maps yes/no correctly;
- training on a sufficiently sized deterministic synthetic dataset returns one prediction per test row and finite numeric MAE, RMSE, and R2 values;
- repeated training with the same input yields the same split/predictions; and
- fewer than 10 cleaned rows raises a clear error.

Do not add low-value tests for pandas, scikit-learn internals, trivial constants, or every possible malformed value.

### Automated verification commands

From the repository root, all must succeed:

```bash
python -m pip install -r requirements.txt
python -m black --check src tests
python -m flake8 src tests
python -m pytest -q
python -m src.main --data data/insurance.csv --output-dir outputs
```

Confirm that all three output files exist, contain plausible row counts and finite metrics where required, and can be parsed as CSV/JSON.

Docker acceptance checks:

```bash
docker build -t insurance-analysis .
docker run --rm insurance-analysis
docker run --rm insurance-analysis python -m pytest -q
```

The default container run must execute the analysis/model script, not merely print a placeholder.

### Manual smoke test and independent Tester

After the Builder conversation ends, the repository owner should independently run the main script, inspect the console output and generated files, and record the date, commands, key observed values, and pass/fail result in the README. This is the manual smoke test; Builder output alone does not satisfy it.

Then start another new AI conversation for the Tester. Give the Tester the repository and this plan, but ask for an independent review rather than implementation. The Tester should rerun formatting, linting, pytest, the CLI, and both Docker commands; compare implementation behavior with this plan; inspect edge-case handling and README accuracy; and report findings by severity. Any fixes should be documented and reverified. Save Architect, Builder, and Tester transcripts under `docs/transcripts/` according to course submission rules.

## 7. README evidence checklist

The final README must include:

- Repository B Option 2 and the project purpose;
- the final directory tree and brief module responsibilities;
- Python 3.11 installation and usage commands;
- pytest, Black, and flake8 commands;
- Docker build, default run, and containerized-test commands;
- a short interpretation of analysis/model results and model limitations;
- actual manual smoke-test commands and observations, not predicted results;
- distinct Architect, Builder, owner/manual-verification, and Tester contributions;
- at least one AI recommendation the owner accepted;
- at least one AI recommendation the owner changed or rejected, with reasoning; and
- how the owner independently verified the result.

## 8. Owner review checkpoints

Before giving this plan to the Builder, the owner should explicitly accept, modify, or reject at least two of these recommendations in the transcript:

1. **Missing data policy — recommended: fail fast.** Challenge whether required-field nulls should raise an error or whether numeric imputation is educationally useful. Fail-fast behavior is more transparent for this known, normally complete dataset; imputation would require a documented train-only strategy to avoid leakage.
2. **Feature scope — recommended: four requested features only.** Challenge whether `sex` and `region` should also be one-hot encoded. Keeping only age, BMI, children, and smoker makes the improvement over the previous model easy to explain; adding categories would require more preprocessing and interpretation. The simplified plan intentionally excludes them.
3. **Generated artifact policy — recommended: regenerate, do not commit.** Challenge whether `outputs/*.json` and `outputs/*.csv` should be Git-ignored or committed as visible evidence. Ignoring avoids stale results; committing makes GitHub review easier. In either case, actual verified key results belong in the README.
4. **Minimum training size — recommended: reject fewer than 10 rows.** Challenge whether 10 is an appropriate teaching-oriented guard or whether the code should calculate a minimum based only on split feasibility. The threshold is intentionally simple but somewhat arbitrary.

Record the owner's decisions before implementation so the Builder follows the approved choices and the final README can cite a genuine accepted and changed/rejected AI recommendation.
