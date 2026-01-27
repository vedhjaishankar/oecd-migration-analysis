# Global Migration & Labor Integration Analysis

## Project Overview
This project provides a comprehensive analytical framework for evaluating global migration trends and labor market integration using multi-dimensional OECD datasets[cite: 31]. It automates the ingestion, cleaning, and visualization of international labor mobility data to quantify parity across demographic cohorts.

The system utilizes a modular Python architecture to transform raw JSON API responses into actionable insights and structured exports[cite: 33].

## Modular Architecture
The repository is structured to ensure scalability and reproducibility:
* **`data_loader.py`**: A specialized module for automating the ingestion of multi-dimensional OECD datasets. It handles API requests, standardizes ISCED education levels, and cleanses migration categories across disparate international sources.
* **`analysis_functions.py`**: Contains the core logic for calculating integration KPIs, such as native vs. foreign-born employment gaps.
* **`vedh_jaishankar_study6.ipynb`**: The primary research notebook that orchestrates the workflow, performs exploratory data analysis (EDA), and generates time-series visualizations.

## Technical Stack
* **Language:** Python 
* **Libraries:** Pandas, Seaborn, Matplotlib 
* **API:** OECD Data API (JSON)
* **Methodology:** Descriptive Statistics, Time-Series Analysis, KPI Engineering

## Getting Started
1. **Requirements:** Install dependencies via `pip install pandas seaborn matplotlib requests`.
2. **Execution:** Run the `vedh_jaishankar_study6.ipynb` notebook to initiate the full data pipeline from extraction to visualization.
3. **Data Exports:** The workflow automatically processes raw responses into structured CSV exports for external stakeholder analysis.
