
from datetime import datetime

from airflow.decorators import dag, task
from include import ml_pipeline


@dag(
    dag_id="weekly_model_comparison",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
)
def weekly_model_comparison():

    @task
    def preprocess_data():
        return ml_pipeline.preprocess_data()

    @task
    def train_baseline(data_path):
        return ml_pipeline.train_baseline(data_path)

    @task
    def train_candidate(data_path):
        return ml_pipeline.train_candidate(data_path)

    @task
    def evaluate_models(baseline_path, candidate_path):
        return ml_pipeline.evaluate_models(
            baseline_path,
            candidate_path
        )

    # Execute the workflow using TaskFlow dependencies
    data_path = preprocess_data()

    baseline_path = train_baseline(data_path)
    candidate_path = train_candidate(data_path)

    evaluate_models(baseline_path, candidate_path)


weekly_model_comparison()
