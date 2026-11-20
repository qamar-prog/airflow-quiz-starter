# Weekly ML Model Comparison DAG

## Project Overview

This project implements a weekly Machine Learning workflow using Apache Airflow and the TaskFlow API.

The workflow uses the Breast Cancer dataset and trains two machine learning models:

1. Logistic Regression (Baseline Model)
2. Random Forest (Candidate Model)

After both models finish training, Airflow executes an evaluation task to compare their performance.

## Project Structure

```text
airflow-quiz-starter/
│
├── dags/
│   └── weekly_model_comparison.py
│
├── include/
│   └── ml_pipeline.py
│
├── plugins/
│
├── Dockerfile
├── requirements.txt
├── packages.txt
└── README.md
```

## Workflow Architecture

```text
              preprocess_data
                     |
              ┌──────┴──────┐
              |             |
        train_baseline  train_candidate
              |             |
              └──────┬──────┘
                     |
              evaluate_models
```

## DAG Configuration

| Configuration | Value |
|---|---|
| DAG ID | weekly_model_comparison |
| Schedule | @weekly |
| Start Date | 2026-01-01 |
| Catchup | False |
| Number of Tasks | 4 |
| Framework | Apache Airflow |
| Runtime | Astro Runtime 12.1.1 |

## Tasks Description

### 1. preprocess_data
Prepares and loads the Breast Cancer dataset.

### 2. train_baseline
Trains the Logistic Regression baseline model using the prepared dataset.

### 3. train_candidate
Trains the Random Forest candidate model using the same prepared dataset.

### 4. evaluate_models
Compares both trained models and displays their evaluation metrics.

The evaluation task runs only after both training tasks have completed.

## Task Dependencies

- preprocess_data → train_baseline
- preprocess_data → train_candidate
- train_baseline → evaluate_models
- train_candidate → evaluate_models

The two training tasks execute independently and can run in parallel.

Dependencies are created using TaskFlow function arguments without using `>>`, `set_upstream()`, or `set_downstream()`.

## Requirements

- Docker Desktop
- Astronomer CLI
- Apache Airflow
- Python
- Astro Runtime 12.1.1

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/qamar-prog/airflow-quiz-starter.git
cd airflow-quiz-starter
```

### 2. Start Docker Desktop

Make sure Docker Desktop is running.

### 3. Start the Airflow project

```bash
astro dev start
```

### 4. Open Airflow UI

Visit:

http://localhost:8080

Login credentials:

- Username: admin
- Password: admin

### 5. Run the DAG

1. Find `weekly_model_comparison` in the DAG list.
2. Unpause the DAG.
3. Click Trigger DAG.
4. Open the manual DAG run.
5. Monitor all four tasks until they succeed.

### 6. View evaluation logs

Open the `evaluate_models` task and select Logs to view the comparison metrics for Logistic Regression and Random Forest.

The final log should display:

```text
Evaluation complete
```

## Author

**Qamar Abbas**

BS Artificial Intelligence

FAST NUCES, Chiniot-Faisalabad Campus

## License

This project was created for academic and educational purposes.
