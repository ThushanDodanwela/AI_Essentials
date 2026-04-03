import argparse
import json
from pathlib import Path

import requests


def post(scoring_uri: str, api_key: str, payload: dict) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    r = requests.post(scoring_uri, headers=headers, data=json.dumps(payload), timeout=60)
    try:
        return {"status_code": r.status_code, "json": r.json()}
    except Exception:
        return {"status_code": r.status_code, "text": r.text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-uri", required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    samples_path = here / "outputs" / "sample_payloads.json"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Sample payloads not found at {samples_path}. Run train_local.py first."
        )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    healthy_payload = {"input_data": samples["healthy"]}
    critical_payload = {"input_data": samples["critical"]}

    print("Sending Healthy Machine payload...")
    print(post(args.scoring_uri, args.api_key, healthy_payload))

    print("Sending Critical Failure payload...")
    print(post(args.scoring_uri, args.api_key, critical_payload))


if __name__ == "__main__":
    main()

