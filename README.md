# Soccer Predictions

This repository automates the generation of football match predictions for the
top five European leagues (Premier League, Ligue 1, Bundesliga, Serie A, La Liga)
based on a set of back‑tested rules and SofaScore team ratings.  When run, the
script fetches the upcoming weekend's fixtures, retrieves average team ratings
from SofaScore, applies the most profitable rules from
`complete_predictive_rules_summary.csv` and outputs a CSV containing only the
matches that satisfy a profitable rule.

## Prerequisites

The script requires Python 3.8+ and the dependencies listed in
`soccer_predictions/requirements.txt`.  These are installed automatically by
the GitHub Actions workflow.

## Usage

Locally you can generate predictions by running:

```bash
pip install -r soccer_predictions/requirements.txt
python soccer_predictions/predict.py --output my_predictions.csv
```

This will create a CSV file `my_predictions.csv` in the current directory
containing predictions for the next weekend.

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow that can be triggered manually
from the Actions tab.  When you run the workflow it will:

1. Check out the repository.
2. Install Python and dependencies.
3. Execute the prediction script.
4. Upload the generated CSV file as a build artifact.

You can optionally specify a custom output file name via the `output_name`
input when dispatching the workflow.  The resulting CSV will be named
`<output_name>.csv`.

## Data Sources

Fixtures are scraped from [worldfootball.net](https://www.worldfootball.net/)
and ratings are obtained via publicly accessible endpoints on
[SofaScore](https://www.sofascore.com/).  The predictive rules and their
performance metrics are provided in `complete_predictive_rules_summary.csv`.