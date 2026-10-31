from sagemaker import image_uris
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model

ROLE = "arn:aws:iam::818651034411:role/sagemaker-ml-role"
BUCKET = "ali886-mlops-data-2026"
sess = PipelineSession()

# ---- Parameters (har run pe badal sakte ho) ----
input_data = ParameterString(name="InputData", default_value=f"s3://{BUCKET}/data/raw/iris.csv")
acc_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.90)

# ---- Step 1: Preprocess ----
pre = SKLearnProcessor(framework_version="1.4-2", role=ROLE, instance_type="ml.t3.medium",
                       instance_count=1, base_job_name="iris-pre", sagemaker_session=sess)
step_pre = ProcessingStep(name="PreprocessIris", step_args=pre.run(
    code="preprocess.py",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
             ProcessingOutput(output_name="test", source="/opt/ml/processing/test")],
))

# ---- Step 2: Train ----
est = PyTorch(entry_point="train.py", source_dir="train_src", role=ROLE,
              instance_type="ml.g6.xlarge", instance_count=1,
              framework_version="2.3", py_version="py311",
              output_path=f"s3://{BUCKET}/models/pipeline/", base_job_name="iris-pipe-train",
              max_run=900, sagemaker_session=sess)
step_train = TrainingStep(name="TrainIris", step_args=est.fit(inputs={
    "train": TrainingInput(s3_data=step_pre.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                           content_type="text/csv")
}))

# ---- Step 3: Evaluate ----
ev = SKLearnProcessor(framework_version="1.4-2", role=ROLE, instance_type="ml.t3.medium",
                      instance_count=1, base_job_name="iris-eval", sagemaker_session=sess)
report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
step_eval = ProcessingStep(name="EvaluateIris", property_files=[report], step_args=ev.run(
    code="evaluate.py",
    inputs=[ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                            destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_pre.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                            destination="/opt/ml/processing/test")],
    outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
))

# ---- Step 5a: Register (agar pass) ----
image = image_uris.retrieve("sklearn", region=sess.boto_region_name, version="1.4-2",
                            py_version="py3", instance_type="ml.c5.xlarge")
model = Model(image_uri=image, model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
              role=ROLE, sagemaker_session=sess,
              env={"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"})
step_register = ModelStep(name="RegisterIris", step_args=model.register(
    content_types=["application/json"], response_types=["application/json"],
    inference_instances=["ml.c5.xlarge"], transform_instances=["ml.c5.xlarge"],
    model_package_group_name="iris-models", approval_status="PendingManualApproval",
))

# ---- Step 5b: Fail (agar fail) ----
step_fail = FailStep(name="AccuracyTooLow",
                     error_message=Join(on=" ", values=["Accuracy is below threshold", acc_threshold]))

# ---- Step 4: Condition ----
step_cond = ConditionStep(
    name="CheckAccuracy",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=step_eval.name, property_file=report, json_path="metrics.accuracy.value"),
        right=acc_threshold)],
    if_steps=[step_register],
    else_steps=[step_fail],
)

# ---- Pipeline banao aur chalao ----
pipeline = Pipeline(name="iris-pipeline", parameters=[input_data, acc_threshold],
                    steps=[step_pre, step_train, step_eval, step_cond], sagemaker_session=sess)
pipeline.upsert(role_arn=ROLE)
execution = pipeline.start()
print("\nPIPELINE STARTED:", execution.arn)
