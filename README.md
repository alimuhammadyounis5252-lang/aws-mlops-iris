# AWS MLOps: Iris Classifier (end-to-end)

Production-style MLOps system on AWS, built step by step.

## Architecture
- SageMaker Pipeline: Preprocess -> Train -> Evaluate -> Accuracy gate (>= 0.90) -> Model Registry (manual approval)
- Event-driven retraining: new data in S3 (data/incoming/) -> EventBridge -> Lambda -> pipeline run
- Monitoring: PSI drift metric -> CloudWatch Alarm -> SNS email, plus a CloudWatch dashboard

## Repo layout
| Path | What |
|---|---|
| `pipeline/` | SageMaker Pipeline definition + preprocess / train / evaluate scripts |
| `pipeline/tests/` | 13 pytest tests: data validation, leakage, quality gate, behaviour, invariance, API contract, dependency pinning |
| `trigger/` | Lambda + EventBridge for event-driven retraining |
| `monitor/` | PSI drift check publishing to CloudWatch + dashboard definition |

## AWS services
S3, IAM, EC2, SageMaker (Training, Processing, Pipelines, Model Registry, Endpoints, HPO), Lambda, EventBridge, CloudWatch, SNS

## Lessons learned
- Pin library versions: train/serve skew (sklearn 1.5.0 vs 1.4.2) broke endpoint deployment
- Serverless endpoints have a read-only filesystem; containers that pip-install at startup fail
- Tune on validation data, not the test set (avoids leakage)
- Tests should check outcomes (accuracy), not settings (hyperparameters)
