# Machine Learning Portfolio

This repository contains machine learning projects and experiments, primarily focused on regression analysis using tennis statistics data.

## Project Structure

- `data_csv/tennis_stats.csv`  
  Raw dataset containing tennis player statistics and performance metrics.

- `scripts/linear_and_multiple_regressions.py`  
  Main script for performing exploratory data analysis and building linear and multiple regression models on the tennis dataset.  
  **Features:**
  - Loads and inspects the tennis statistics dataset.
  - Generates scatter plots to visualize relationships between features and winnings.
  - Performs single, two-feature, and multiple-feature linear regressions to predict player winnings.
  - Prints model coefficients, intercepts, and scores for evaluation.

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) as the package manager for fast and reproducible Python environments.

### Install dependencies

```sh
uv sync
```