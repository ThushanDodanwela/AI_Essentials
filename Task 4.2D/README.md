# Task 4.2D — Azure ML SDK v2 (AI4I 2020)

This folder contains everything required for the SIT788 Task 4.2D deliverables:

- Local training on **AI4I 2020 Predictive Maintenance** data (`ai4i2020.csv`)
- At least one engineered feature (`temp_delta_k`)
- Azure ML SDK v2 asset registration (Model + Environment)
- Managed Online Endpoint deployment (no legacy ACI)
- Python REST API testing with two payloads (healthy vs critical)

## 1) Local training (creates `outputs/model.joblib`)

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python train_local.py
```

Outputs:
- `outputs/model.joblib`
- `outputs/sample_payloads.json`

Optional (for your report discussion of engineered feature importance):

```powershell
python feature_importance.py
```

## 2) Deploy to Azure ML (SDK v2)

Prereqs:
- You can sign in: `az login`
- Your workspace exists in Azure ML Studio

Run:

```powershell
python deploy_azureml.py `
  --subscription-id "<SUB_ID>" `
  --resource-group "<RG>" `
  --workspace "<WS>" `
  --endpoint-name "ai4i-pm-endpoint-<yourname>"
```

The script prints:
- scoring URI
- primary key

## 3) Test the live endpoint (REST)

```powershell
python test_endpoint.py --scoring-uri "<SCORING_URI>" --api-key "<PRIMARY_KEY>"
```

## Why `temp_delta_k` matters

`temp_delta_k = Process temperature [K] - Air temperature [K]`

If the process/workpiece runs much hotter than the surrounding air, the machine is under higher **thermal stress**.
Thermal stress is a realistic driver of failure because it accelerates wear, degrades lubrication, and can push parts out of tolerance.

