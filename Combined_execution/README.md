# Combined Execution — Final Challenge

## Folder placement
Put this folder beside `Day-1` and `Day-2`:

- `../Day-1/day1_log.csv`
- `../Day-2/Data/day2_log.csv`

The dashboard also checks `day1_data.csv` and `day2_data.csv` alternatives listed in `app.py`.

## VS Code extensions
Install Microsoft Python and Pylance. Jupyter is optional.

## Install and run
Open a terminal in this folder:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, run:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Outputs
`outputs/combined_dataset.csv`, `outputs/best_machine_state_model.pkl`, `outputs/event_log.csv`.

## Demo PIN
`241432` in app.py. Change it before sharing. This app only simulates actions; it does not control real equipment.

## Dataset requirements
Both CSVs must have `temperature`, `pressure`, `vibration`, `current`, `rpm`, and `state` columns to contribute to model training. Column aliases such as `temp`, `motor_rpm`, `label`, and `status` are normalized. If Day-1's file is a system event log instead of labelled sensor data, it will be excluded from ML training and reported in the dashboard.
