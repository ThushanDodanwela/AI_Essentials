import argparse
import time
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential


def wait_for_endpoint(ml_client: MLClient, endpoint_name: str) -> None:
    while True:
        ep = ml_client.online_endpoints.get(endpoint_name)
        state = getattr(ep, "provisioning_state", None)
        print(f"Endpoint state: {state}")
        if state in ("Succeeded", "Failed", "Canceled"):
            if state != "Succeeded":
                raise RuntimeError(f"Endpoint provisioning ended in state={state}")
            return
        time.sleep(10)


def wait_for_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    while True:
        dep = ml_client.online_deployments.get(endpoint_name=endpoint_name, name=deployment_name)
        state = getattr(dep, "provisioning_state", None)
        print(f"Deployment state: {state}")
        if state in ("Succeeded", "Failed", "Canceled"):
            if state != "Succeeded":
                raise RuntimeError(f"Deployment provisioning ended in state={state}")
            return
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--deployment-name", default="blue")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    model_file = here / "outputs" / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found at {model_file}. Run train_local.py first."
        )

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    # 1) Register environment (versioned asset)
    env = Environment(
        name="ai4i-pm-env",
        description="Conda env for AI4I predictive maintenance endpoint",
        conda_file=str(here / "conda.yml"),
        image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py310-cpu-inference:latest",
    )
    env = ml_client.environments.create_or_update(env)
    print(f"Registered environment: {env.name}:{env.version}")

    # 2) Register model (versioned asset)
    model = Model(
        name="ai4i-pm-rf",
        path=str(model_file),
        description="RandomForest pipeline for AI4I 2020 predictive maintenance",
        type="custom_model",
    )
    model = ml_client.models.create_or_update(model)
    print(f"Registered model: {model.name}:{model.version}")

    # 3) Create (or update) managed online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="Managed online endpoint for AI4I predictive maintenance",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    wait_for_endpoint(ml_client, args.endpoint_name)

    # 4) Create managed deployment
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(code=str(here), scoring_script="score.py"),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    wait_for_deployment(ml_client, args.endpoint_name, args.deployment_name)

    # 5) Route 100% traffic to this deployment
    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {args.deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("Traffic set to 100%.")

    # 6) Print endpoint details + keys for testing
    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    keys = ml_client.online_endpoints.get_keys(name=args.endpoint_name)
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"Primary key: {keys.primary_key}")


if __name__ == "__main__":
    main()

