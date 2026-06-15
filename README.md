# Cancer Transcription Factor Discovery

A machine learning pipeline that predicts cancer cell growth from protein and gene expression data and identifies which transcription factors are most strongly associated with cancer. Built with a Random Forest classifier and deployed as an interactive Streamlit dashboard.

---

## Project summary

Transcription factors (TFs) are proteins that control which genes get switched on or off in a cell. When TFs malfunction or are overexpressed, they can drive uncontrolled cell division, the defining characteristic of cancer. Identifying which TFs are most predictive of cancer versus normal cell states is a foundational step in understanding cancer biology and in prioritizing drug targets.

This project builds a complete end-to-end pipeline:

1. Combined protein expression (80 features) and gene expression (68 features) into a single feature matrix across 1,200 cells
2. Trained a Random Forest classifier to distinguish cancer from normal cells
3. Extracted and ranked transcription factor importance using both Gini impurity (MDI) and permutation importance
4. Deployed the model as a Streamlit dashboard with CSV upload, live predictions, and interactive visualizations

---

## Repository structure

```
cancer-tf-discovery/
|-- app.py                    Streamlit dashboard
|-- cancer_tf_discovery.ipynb Full step-by-step analysis notebook
|-- requirements.txt          Python dependencies
|-- data/
|   |-- gene_expression.csv   Gene expression matrix (1200 x 69)
|   |-- protein_expression.csv Protein expression matrix (1200 x 81)
|   |-- cell_labels.csv       Binary cancer/normal labels
|   |-- tf_annotation.csv     Transcription factor gene symbol list
|-- outputs/
|   |-- model_bundle.pkl      Saved model, scaler, and feature names
|   |-- tf_ranking_results.csv Ranked TF importance table
|   |-- tf_importance_mdi.png  MDI importance bar chart
|   |-- tf_importance_perm.png Permutation importance bar chart
|   |-- tf_violin_plots.png    Expression distribution per TF
```

---

## Results

| Metric | Value |
|---|---|
| Accuracy (test set) | 91.4% |
| AUC-ROC | 0.957 |
| Cross-validation AUC | 0.953 +/- 0.008 |
| Total features | 148 (80 protein + 68 gene) |
| Training samples | 960 |
| Test samples | 240 |

### Top transcription factors by permutation importance

| Rank | TF | Protein family | Cancer association |
|---|---|---|---|
| 1 | TP53 | p53 | High |
| 2 | MYC | bHLH | High |
| 3 | HIF1A | bHLH | High |
| 4 | FOXM1 | Forkhead | High |
| 5 | E2F1 | E2F | High |
| 6 | NFKB1 | Rel | Moderate |
| 7 | SP1 | SP/KLF | Moderate |
| 8 | STAT3 | STAT | Moderate |

TP53 and MYC rank highest because they are master regulators of cell cycle arrest and proliferation respectively. TP53 is mutated in over 50% of all human cancers, and MYC is amplified or overexpressed in approximately 70%. Their dominance in the feature importance ranking is biologically consistent with known cancer mechanisms.

---

## Dashboard features

The Streamlit dashboard (`app.py`) has four sections:

**Overview and model performance** shows key metrics (accuracy, AUC), the ROC curve, and the confusion matrix on the full dataset.

**Transcription factor ranking** displays an interactive importance chart with a slider to control how many TFs are shown, plus a downloadable ranked table.

**Expression analysis** compares mean TF expression between cancer and normal cells with a grouped bar chart, and provides a violin plot for any individual TF selected from a dropdown.

**Predict new cells** accepts uploaded protein and gene expression CSVs, returns cancer/normal predictions with probability scores, shows the distribution of scores across all submitted cells, and provides a downloadable results CSV.

---

## How to run locally

```bash
git clone https://github.com/tracyvuleze2-sketch/cancer-tf-discovery
cd cancer-tf-discovery

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. All four CSV data files and `model_bundle.pkl` must be in the same directory as `app.py`.

---

## Notebook walkthrough

The Jupyter notebook `cancer_tf_discovery.ipynb` walks through the full pipeline in 11 steps:

1. Import libraries
2. Load and inspect all four data files
3. Exploratory data analysis: distribution plots, fold change table, correlation heatmap
4. Build the merged feature matrix with log1p normalization
5. Train/test split (80/20 stratified) and StandardScaler
6. Train the Random Forest with `class_weight='balanced'`
7. 5-fold stratified cross-validation
8. Evaluate on the held-out test set: classification report, confusion matrix, ROC curve
9. Rank TFs using MDI importance and permutation importance
10. Visualize: bar charts, error bars, violin plots
11. Export: `tf_ranking_results.csv` and `all_feature_importance.csv`

---

## Biological context

The transcription factors in this dataset were selected from the TFdb and JASPAR databases. Each of the top-ranked TFs has a well-established role in cancer biology:

**TP53** is the most frequently mutated gene in all human cancers. It acts as the cell's stress sensor: when DNA is damaged, TP53 activates genes that pause the cell cycle or trigger apoptosis. In cancer, TP53 is almost always inactivated, allowing cells with damaged DNA to keep dividing.

**MYC** forces cells to grow and divide at an abnormally high rate, reprograms metabolism toward aerobic glycolysis (the Warburg effect), and suppresses immune surveillance. It is amplified in approximately 70% of cancers.

**HIF1A** is constitutively active in most tumors because oncogenic signaling from RAS and PI3K stabilizes it even under normal oxygen. It drives tumor vascularization and glycolytic metabolism.

**FOXM1** overexpression accelerates the G2/M cell cycle transition, promotes chromosomal instability, and is associated with poor prognosis across multiple cancer types.

---

## Technologies

- Python 3.10+
- scikit-learn: Random Forest, permutation importance, cross-validation
- pandas and numpy: data loading, feature engineering, log1p normalization
- matplotlib and seaborn: visualization
- Streamlit: interactive dashboard
- joblib: model serialization

---

## Dataset note

The data used in this project is synthetically generated to reproduce realistic biological signal patterns observed in cancer genomics studies. The gene and protein expression values follow log-normal distributions with cancer-specific upregulation parameters drawn from published differential expression studies. This allows full reproducibility without requiring access to restricted patient data.

To use this pipeline on real data, replace the four CSV files with your own expression matrices and label file. The column format is identical: rows are cells, columns are features, index column is `cell_id`.

---

## Author

Tracy Vuleze | Data Science student, Moringa School (DSF-FT16)

GitHub: [tracyvuleze2-sketch](https://github.com/tracyvuleze2-sketch)

---

## License

MIT
