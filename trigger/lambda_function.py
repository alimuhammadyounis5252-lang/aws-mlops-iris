import json, boto3

sm = boto3.client("sagemaker")
PIPELINE = "iris-pipeline"

def lambda_handler(event, context):
    print("EVENT:", json.dumps(event))
    bucket = event["detail"]["bucket"]["name"]
    key = event["detail"]["object"]["key"]

    if not key.endswith("iris.csv"):
        print("Skipping, not an iris.csv file:", key)
        return {"started": False}

    s3_uri = f"s3://{bucket}/{key}"
    resp = sm.start_pipeline_execution(
        PipelineName=PIPELINE,
        PipelineExecutionDisplayName="auto-" + key.split("/")[-2],
        PipelineParameters=[{"Name": "InputData", "Value": s3_uri}],
    )
    print("Started pipeline for", s3_uri, "->", resp["PipelineExecutionArn"])
    return {"started": True, "arn": resp["PipelineExecutionArn"]}
