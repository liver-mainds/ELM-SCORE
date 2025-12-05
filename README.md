# ELM Score: Pan-Elastography Machine Learning Model for CSPH Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Official implementation of the ELM Score for non-invasive prediction of Clinically Significant Portal Hypertension (CSPH)**

> **Paper**: "The ELasto-ML (ELM) Score: Validation of a Pan-Elastography Machine-Learning Model for Non-Invasive Prediction of Clinically Significant Portal Hypertension in Compensated Advanced Chronic Liver Disease"

---

## Overview

The **ELM Score** is a validated machine learning model that predicts the presence of Clinically Significant Portal Hypertension (CSPH, defined as HVPG ≥10 mmHg) using non-invasive parameters across multiple elastography platforms.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/liver-mainds/ELM-SCORE.git
cd ELM-SCORE

# Install required packages
pip install -r requirements.txt
```

### Basic Usage

```bash
python elm_score_inference.py \
    --input your_patient_data.csv \
    --model rf_model.pkl \
    --params z_score_params.json \
    --output predictions.csv
```

---

## 📊 Input Data Format

### Required Columns

Your input CSV or Excel file must contain the following columns with **exact names**:

| Column Name | Description | Type | Valid Values |
|-------------|-------------|------|--------------|
| `Age` | Patient age | Integer | Years |
| `Sex` | Patient sex | Integer | 0 = Female, 1 = Male |
| `PLT` | Platelet count | Float | ×10⁹/L |
| `CPT_Score` | Child-Pugh-Turcotte Score | Integer | 5, 6, o 7 |
| `Viral Etiology` | Viral hepatitis etiology | Integer | 0 = No, 1 = Yes (HCV/HBV) |
| `Alcohol Etiology` | Alcohol-associated liver disease | Integer | 0 = No, 1 = Yes (ALD) |
| `Metabolic Etiology` | Metabolic dysfunction-associated steatotic liver disease | Integer | 0 = No, 1 = Yes (MASLD) |
| `Autoimmune Etiology` | Autoimmune etiology | Integer | 0 = No, 1 = Yes |
| `Other/mixed Etiology` | Other/mixed etiology | Integer | 0 = No, 1 = Yes |
| `LS` | Liver Stiffness Measurement | Float | kPa |
| `SS` | Spleen Stiffness Measurement | Float | kPa |
| `Elastography Modality` | Elastography device | String | 'VCTE-50Hz', 'VCTE-100Hz', 'P-SWE', '2D-SWE' |

### Example Input File (CSV)

```csv
Age,Sex,PLT,CPT_Score,Viral Etiology,Alcohol Etiology,Metabolic Etiology,Autoimmune Etiology,Other/mixed Etiology,LS,SS,Elastography Modality
62,1,95,6,1,1,0,0,0,28.5,55.2,VCTE-50Hz
55,0,145,5,0,0,0,0,0,18.3,35.7,2D-SWE
58,1,112,5,1,0,1,0,0,22.1,48.9,P-SWE
```

### Important Notes

1. **Column names are case-sensitive** and must match exactly as shown above
2. **Stiffness values (LS, SS)** must be in **kilopascals (kPa)**
   - If your values are in m/s, convert them using: `kPa = 3 × 1050 × (m/s)² / 1000`
3. **Elastography Modality values** are standardized:
   - `VCTE-50Hz`: FibroScan with 50 Hz probe
   - `VCTE-100Hz`: FibroScan with 100 Hz probe
   - `P-SWE`: Point Shear Wave Elastography (ElastPQ or VTQ)
   - `2D-SWE`: Two-Dimensional Shear Wave Elastography (Aixplorer)
4. **CPT_Score** should only include patients with scores of 5, 6, or 7 (compensated disease)
5. **Missing values**: The Random Forest model can handle missing values using surrogate splits, but performance may be reduced

---

## 📈 Output Format

The script generates a CSV file with all original columns plus the following:

| Column Name | Description | Values |
|-------------|-------------|--------|
| `LS_zscore` | Z-score normalized liver stiffness | Float |
| `SS_zscore` | Z-score normalized spleen stiffness | Float |
| `ELM_Score` | Predicted probability of CSPH | 0.0 - 1.0 |
| `CSPH_Predicted` | Binary prediction | 0 = No CSPH, 1 = CSPH |
| `Risk_Category` | Clinical risk stratification | See below |

### Risk Categories

- **Rule-Out (Low Risk)**: ELM Score ≤ 0.45
  - Low probability of CSPH
  - NPV: 0.90 (95% CI: 0.84-0.94)
  - Consider deferring invasive HVPG measurement

- **Rule-In (High Risk)**: ELM Score ≥ 0.60
  - High probability of CSPH
  - PPV: 0.96 (95% CI: 0.92-0.98)
  - Consider starting non-selective β-blocker therapy

- **Gray Zone**: ELM Score 0.45-0.60
  - Indeterminate risk
  - Consider confirmatory testing (HVPG or endoscopy)

---

## 🔧 Custom Usage

### Command-Line Arguments

```bash
python elm_score_inference.py --help
```

```
Options:
  -i, --input INPUT      Path to input CSV or Excel file (required)
  -m, --model MODEL      Path to trained Random Forest model .pkl file (default: rf_model.pkl)
  -p, --params PARAMS    Path to z-score parameters .json file (default: z_score_params.json)
  -o, --output OUTPUT    Path to output CSV file (default: input_predictions.csv)
```

### Python API Usage

```python
from elm_score_inference import ELMScoreInference

# Initialize inference engine
engine = ELMScoreInference(
    z_score_params_path='z_score_params.json',
    model_path='rf_model.joblib'
)

# Run inference on a file
results = engine.run_inference(
    input_file='patient_data.csv',
    output_file='predictions.csv'
)

# Access predictions
print(f"Mean ELM Score: {results['ELM_Score'].mean():.3f}")
print(f"High-risk patients: {(results['Risk_Category'] == 'Rule-In (High Risk)').sum()}")
```

### Custom Workflow

```python
import pandas as pd
from elm_score_inference import ELMScoreInference

# Initialize
engine = ELMScoreInference('z_score_params.json', 'rf_model.pkl')

# Load your data
df = pd.read_csv('my_data.csv')

# Validate and prepare
engine.validate_columns(df)
df = engine.validate_manufacturer(df)
df = engine.normalize_stiffness(df)

# Prepare features
X = engine.prepare_features(df)

# Get predictions
y_pred_class, y_pred_proba = engine.predict(X)

# Add results to dataframe
df['ELM_Score'] = y_pred_proba
df['CSPH_Predicted'] = y_pred_class
df['Risk_Category'] = engine.interpret_score(y_pred_proba)
```

---

## 📦 Repository Contents

```
ELM-SCORE/
├── elm_score_inference.py      # Main inference script
├── rf_model.pkl                 # Trained Random Forest model
├── z_score_params.json          # Z-score normalization parameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
└── examples/
    ├── example_input.csv        # Example input data
    └── example_output.csv       # Example predictions
```

---

## 🔬 Model Details

### Architecture

- **Model Type**: Random Forest Classifier
- **Hyperparameters**:
  - `n_estimators`: 125
  - `criterion`: 'entropy'
  - `max_depth`: 6
  - `min_samples_split`: 22
  - `min_samples_leaf`: 3
  - `max_features`: 'sqrt'
  - `random_state`: 42

### Feature Importance

Based on both impurity-based and mutual information analyses:

1. **Spleen Stiffness (SS_zscore)** - Most important
2. **Liver Stiffness (LS_zscore)** - Second most important
3. **Platelet Count (PLT)** - Third most important
4. Age, CPT Score, Sex, Viral Etiology - Supporting features

### Training Data

- **Training cohort**: 943 patients from 17 centers
- **Internal validation**: 150 patients
- **External validation**: 342 patients from 7 independent centers
- **Elastography modalities**: VCTE (50Hz and 100Hz), 2D-SWE, pSWE
- **Etiologies**: Viral (HCV/HBV), ALD, MASLD, Autoimmune, Mixed/Other

### Performance Metrics (External Validation)

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC | 0.91 | - |
| Accuracy | 0.82 | - |
| Sensitivity (Rule-out) | 0.94 | 0.91-0.97 |
| Specificity (Rule-in) | 0.94 | 0.80-0.98 |
| NPV | 0.90 | 0.84-0.94 |
| PPV | 0.96 | 0.92-0.98 |
| Gray Zone | 12.3% | - |

### Comparison with Existing Criteria in the External Validation Cohort

| Criterion | Gray Zone | NPV (Rule-Out) | PPV (Rule-In) |
|-----------|-----------|-----|-----|
| **ELM Score** | **12.3%** | **0.90** | **0.96** |
| Baveno VII | 47.9% | 0.96 | 0.90 |
| Baveno-SSM Dual Cut-Off Model | 38.6% | 0.97 | 0.92 |
| Baveno-SSM Single Cut-Off Model | 19.6% | 0.88 | 0.86 |

---
## ⚠️ Important Disclaimers

### Research Use Only

**This model is currently a research tool and has NOT received regulatory approval for clinical use.**

- ✗ Not FDA approved
- ✗ Not CE marked
- ✗ Not validated for routine clinical decision-making

### Clinical Responsibility

- The model is provided for research and validation purposes only
- Clinical decisions should be made by qualified healthcare professionals
- The model does not replace clinical judgment or established diagnostic procedures
- Always consider the full clinical context when interpreting results

## 🏥 Clinical Applications

### Primary Use Case

**Non-invasive selection of patients for non-selective β-blocker (NSBB) therapy**

- Patients with ELM Score ≥0.60 can be considered for NSBB therapy (e.g., carvedilol)
- Patients with ELM Score ≤0.45 can defer invasive HVPG measurement
- Patients in gray zone (0.45-0.60) should be evaluated with additional testing

### Advantages

1. **Reduces need for invasive HVPG measurement**
2. **Works across all major elastography platforms**
3. **Minimizes indeterminate results**
4. **Captures more high-risk patients** (88.4% vs. 57.6% with Baveno VII)
5. **Easy to implement** in clinical practice

### Limitations

1. **Not validated for decompensated cirrhosis** (CPT Score >7)
2. **Performance may vary** with extreme BMI (>35 kg/m²)
3. **SSM availability** may be limited in some centers
4. **Requires regulatory approval** for clinical use (research tool only)

---

## 📚 Citation

If you use the ELM Score in your research, please cite:

```bibtex
@article{baveno2025elm,
  title={The ELasto-ML (ELM) Score: Validation of a Pan-Elastography Machine-Learning Model for Non-Invasive Prediction of Clinically Significant Portal Hypertension},
  author={Giuffrè, Mauro and Kresevic, Simone and ...},
  journal={...},
  year={...},
  note={A Baveno Cooperation Study, EASL Consortium}
}
```

**Web-based calculator**: https://elmscore.com  
**Code repository**: https://github.com/liver-mainds/ELM-SCORE

---



## 🛠️ Requirements

### Python Version

Python 3.8 or higher

### Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Testing

Run the test suite to verify installation:

```bash
python -m pytest tests/
```

Or test with the example data:

```bash
python elm_score_inference.py \
    --input examples/example_input.csv \
    # Model and parameters use default files if not specified:
    # --model rf_model.pkl (default)
    # --params z_score_params.json (default)
```

---

## 🤝 Contributing

We welcome contributions from the community! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- Additional validation cohorts
- Integration with EHR systems
- Performance optimization
- Documentation improvements
- Bug fixes and feature requests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work is part of the **Baveno Cooperation**, an EASL Consortium initiative. We thank:

- All participating centers and investigators
- The patients who contributed data to this study
- The Baveno Cooperation steering committee
- EASL (European Association for the Study of the Liver)

---

## ❓ FAQ

### Q: Can I use this model in my clinical practice?

**A**: No. The model is for research purposes only and requires regulatory approval before clinical implementation.

### Q: What if I don't have Spleen Stiffness Measurement (SSM)?

**A**: The model can technically run with missing SSM values using Random Forest's surrogate splits, but performance will be degraded. We provide this capability on GitHub for research purposes, but recommend obtaining SSM measurements when possible.

### Q: Which elastography device should I use?

**A**: The model works with all three major platforms (VCTE-50Hz, VCTE-100Hz, 2D-SWE, pSWE). Use whichever device is available at your center. The model's z-score harmonization accounts for inter-device variability.

### Q: Can I use this for decompensated cirrhosis?

**A**: No. The model was developed and validated only for compensated advanced chronic liver disease (CPT Score ≤7). Do not use it for decompensated patients.

### Q: How do I handle the gray zone patients?

**A**: Patients with ELM Score between 0.45-0.60 should undergo confirmatory testing with HVPG measurement or upper endoscopy for varices screening, according to local protocols.

### Q: Is the model validated for metabolic dysfunction-associated steatotic liver disease (MASLD)?

**A**: Yes. The external validation cohort included 33.9% MASLD patients, and the model showed excellent performance in this subgroup (gray zone 7.1%).

---

**For questions, issues, or collaborations, please contact the authors directly.**
