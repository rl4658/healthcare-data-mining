# MedSynora DW — Multi-Dimensional Patient Stratification

**CMPE 255 Data Mining Class Project**

## Project Objective

Group ~148,000 patient encounters into clusters based on medical severity, vital signs, and economic impact to identify high-utilization patient segments for healthcare resource optimization.

## Dataset

[MedSynora DW](https://www.kaggle.com/datasets/mebrar21/medsynora-dw) — A synthetic healthcare data warehouse containing 27 relational files with fact tables, dimension tables, and bridge tables implementing a star schema architecture.

## Architecture

- **Star Schema** centered on `FactEncounter`, linked to `DimPatient`, `DimDisease`, `DimDoctor`, `DimInsurance`, `DimRoom`, `DimDate`
- **ETL Pipeline** merges 6 tables into a single Master Analytical View
- **K-Means Clustering** with from-scratch iterative visualization

## Project Structure

```
cmpe255-project/
├── main.py                    # Master orchestrator — runs full pipeline
├── src/
│   ├── etl.py                 # ETL: load & merge 6 tables
│   ├── preprocessing.py       # Imputation, IQR, log-transform, scaling
│   ├── eda.py                 # 7 EDA visualizations
│   ├── kmeans_iterative.py    # Custom K-Means with per-iteration plots
│   ├── evaluation.py          # Elbow, Silhouette, Cluster Heatmap
│   └── plantuml_schema.py     # PlantUML star schema generator
├── MedSynora DW/              # Raw dataset (27 CSV files)
├── output/                    # Generated outputs
│   ├── plots/eda/
│   ├── plots/kmeans/
│   ├── plots/evaluation/
│   └── plantuml/
└── requirements.txt
```

## Setup & Run

```bash
pip install -r requirements.txt
python main.py
```

## Methodology

1. **ETL** — Extract data from 6 tables, transform (calculate age, BMI), load into Master Analytical View
2. **Preprocessing** — Median imputation → IQR outlier removal → Log-transform → StandardScaler
3. **EDA** — Distribution analysis, correlation heatmap, box plots, vital signs panel
4. **K-Means** — From-scratch implementation with iteration-by-iteration E-step/M-step visualization
5. **Evaluation** — Elbow method (WCSS), Silhouette analysis, Cluster profile heatmap

## Mathematical Foundation

- **WCSS Objective**: J = Σᵢ Σ_{x∈Cᵢ} ||x − μᵢ||²
- **Euclidean Distance**: d(x,μ) = √(Σⱼ (xⱼ − μⱼ)²)
- **Centroid Update**: μᵢ = (1/|Cᵢ|) Σ_{x∈Cᵢ} x

## Conclusions
The clustering model successfully stratifies the patient population into meaningful segments based on severity and resource usage. By interpreting these clusters, healthcare administrators can proactively identify high-risk, high-cost patient groups, optimize staffing for severe cases, and allocate preventative care resources to those with escalating vitals before they require emergency intervention.

## Future Work
- **Classification Modeling**: Develop a predictive model (e.g., Random Forest or XGBoost) to classify new, incoming patient encounters into these identified severity clusters in real-time.
- **Time-Series Analysis**: Incorporate temporal sequences of patient visits to predict disease progression and worsening clinical indicators over time.
- **Enhanced Deployment**: Package the analytical pipeline into a containerized REST API (via Flask or FastAPI) or a real-time web dashboard using Streamlit to provide on-demand stratification for hospital triage systems.
