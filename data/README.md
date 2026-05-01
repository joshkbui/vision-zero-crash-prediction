# Data

This directory contains data for the Vision Zero Injury Risk Simulator project.

## Dataset

**Montgomery County Crash Reporting – Drivers Data**

- **Source:** [Montgomery County Open Data Portal](https://data.montgomerycountymd.gov/)
- **Direct link:** Search for "Crash Reporting - Drivers Data" on the portal
- **Format:** CSV
- **Size:** ~201,000 rows
- **License:** Montgomery County Open Data

## Download Instructions

1. Navigate to the [Montgomery County Open Data Portal](https://data.montgomerycountymd.gov/)
2. Search for **"Crash Reporting - Drivers Data"**
3. Click **Export** → **CSV**
4. Save the file to this `data/` directory
5. Rename the file to `Crash_Reporting_Drivers_Data.csv` (or update the path in the notebook)

## Files

| File | Description |
|---|---|
| `Crash_Reporting_Drivers_Data.csv` | Raw crash data (not tracked in git — see .gitignore) |
| `README.md` | This file |

## Notes

- The raw CSV file is **not tracked in this repository** (excluded by `.gitignore`) due to its large size
- All data used in this project is publicly available from Montgomery County
- No PII or sensitive data is included in the features used for modeling
