"""
ELM Score Inference Script
==========================
Script for running inference with the ELM Score Random Forest model
for predicting Clinically Significant Portal Hypertension (CSPH).

Author: LIVER-MAINDS Laboratory
Repository: https://github.com/liver-mainds/ELM-SCORE
Paper: "The ELasto-ML (ELM) Score: Validation of a Pan-Elastography Machine-Learning Model"
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path


class ELMScoreInference:
    """
    ELM Score inference class for CSPH prediction.
    
    This class handles:
    - Loading input data (CSV or Excel)
    - Z-score normalization of elastography measurements
    - Feature validation and preparation
    - Model inference
    - Output generation
    """
    
    REQUIRED_COLUMNS = [
        'Age',           # Patient age in years
        'Sex',           # Sex: 0 = Female, 1 = Male
        'PLT',           # Platelet count (×10⁹/L)
        'CPT_Score',     # Child-Pugh-Turcotte Score (5, 6, or 7)
        'Viral Etiology',  # Viral etiology: 0 = No, 1 = Yes (HCV or HBV)
        'Alcohol Etiology', # Alcohol associated liver disease (: 0 = No, 1 = Yes (HCV or HBV)
        'Metabolic Etiology', # Metabolic dysfunction-associated steatotic liver disease etiology: 0 = No, 1 = Yes (HCV or HBV)
        'Autoimmune Etiology', # Autoimmune etiology: 0 = No, 1 = Yes (HCV or HBV)
        'Other/mixed Etiology', # Other/mixed associated liver disease (: 0 = No, 1 = Yes (HCV or HBV)
        'LS',            # Liver Stiffness Measurement in kPa
        'SS',            # Spleen Stiffness Measurement in kPa
        'Elastography Modality'   # Elastography device: 'VCTE-50Hz', 'VCTE-100Hz', 'P-SWE', or '2D-SWE'
    ]
    
    VALID_MANUFACTURERS = ['VCTE-50Hz', 'VCTE-100Hz', 'P-SWE', '2D-SWE']
    
    # Manufacturer mapping (VCTE-50Hz and VCTE-100Hz share the same normalization parameters)
    MANUFACTURER_MAP = {
        'VCTE-50Hz': 'VCTE',
        'VCTE-100Hz': 'VCTE',
        'P-SWE': 'P-SWE',
        '2D-SWE': '2D-SWE'
    }
    
    def __init__(self, z_score_params_path, model_path):
        """
        Initialize the ELM Score inference engine.
        
        Parameters:
        -----------
        z_score_params_path : str or Path
            Path to the JSON file containing z-score normalization parameters
        model_path : str or Path
            Path to the trained Random Forest model (.pkl file)
        """
        self.z_score_params = self._load_z_score_params(z_score_params_path)
        self.model = self._load_model(model_path)
        
    def _load_z_score_params(self, path):
        """Load z-score normalization parameters from JSON file."""
        try:
            with open(path, 'r') as f:
                params = json.load(f)
            print(f"✓ Successfully loaded z-score parameters from: {path}")
            return params
        except FileNotFoundError:
            print(f"✗ Error: Z-score parameters file not found: {path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"✗ Error: Invalid JSON format in: {path}")
            sys.exit(1)

    def _load_model(self, path):
        """Load the trained Random Forest model."""
        try:
            import joblib
            with open(path, 'rb') as f:
                model = joblib.load(f)
            print(f"✓ Successfully loaded model from: {path}")
            return model
        except FileNotFoundError:
            print(f"✗ Error: Model file not found: {path}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def load_data(self, file_path):
        """
        Load data from CSV or Excel file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the input data file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"✗ Error: Input file not found: {file_path}")
            sys.exit(1)
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                print(f"✗ Error: Unsupported file format. Use .csv, .xlsx, or .xls")
                sys.exit(1)
            
            print(f"✓ Successfully loaded data from: {file_path}")
            print(f"  Found {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"✗ Error loading data file: {e}")
            sys.exit(1)
    
    def validate_columns(self, df):
        """
        Validate that all required columns are present in the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to validate
            
        Returns:
        --------
        bool
            True if validation passes, exits otherwise
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            print(f"\n✗ Error: Missing required columns:")
            for col in missing_cols:
                print(f"  - {col}")
            print(f"\nRequired columns are:")
            for col in self.REQUIRED_COLUMNS:
                print(f"  - {col}")
            print("\nPlease note thet column names are case-sensitive** and must match exactly as shown above.")
            sys.exit(1)
        
        print("✓ All required columns present")
        return True
    
    def validate_manufacturer(self, df):
        """
        Validate manufacturer values and show warnings for invalid entries.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to validate
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with validated manufacturers
        """
        invalid_manufacturers = df[~df['Elastography Modality'].isin(self.VALID_MANUFACTURERS)]
        
        if len(invalid_manufacturers) > 0:
            print(f"\n⚠ Warning: Found {len(invalid_manufacturers)} rows with invalid Elastography Modality values")
            print("Valid modalities are: VCTE-50Hz, VCTE-100Hz, P-SWE, 2D-SWE")
            print(f"Invalid values found: {invalid_manufacturers['Elastography Modality'].unique().tolist()}")
            print("These rows will be skipped during inference.\n")
        
        # Filter to valid manufacturers only
        valid_df = df[df['Elastography Modality'].isin(self.VALID_MANUFACTURERS)].copy()
        print(f"✓ {len(valid_df)} rows with valid elastography modalities")
        
        return valid_df
    
    def normalize_stiffness(self, df):
        """
        Apply z-score normalization to liver and spleen stiffness measurements
        based on manufacturer-specific parameters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with LS and SS columns
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added LS_zscore and SS_zscore columns
        """
        df = df.copy()
        
        # Map manufacturers (VCTE-50Hz and VCTE-100Hz -> VCTE for normalization)
        df['Manufacturer_normalized'] = df['Elastography Modality'].map(self.MANUFACTURER_MAP)
        
        # Initialize z-score columns
        df['LS_zscore'] = np.nan
        df['SS_zscore'] = np.nan
        
        # Apply z-score normalization for each manufacturer
        for manufacturer in df['Manufacturer_normalized'].unique():
            if manufacturer not in self.z_score_params:
                print(f"⚠ Warning: No z-score parameters for manufacturer: {manufacturer}")
                continue
            
            # Get parameters for this manufacturer
            ls_mean = self.z_score_params[manufacturer]['LS']['mean']
            ls_std = self.z_score_params[manufacturer]['LS']['std']
            ss_mean = self.z_score_params[manufacturer]['SS']['mean']
            ss_std = self.z_score_params[manufacturer]['SS']['std']
            
            # Apply z-score transformation
            mask = df['Manufacturer_normalized'] == manufacturer
            df.loc[mask, 'LS_zscore'] = (df.loc[mask, 'LS'] - ls_mean) / ls_std
            df.loc[mask, 'SS_zscore'] = (df.loc[mask, 'SS'] - ss_mean) / ss_std
        
        print("✓ Z-score normalization completed")
        print(f"  LS_zscore range: [{df['LS_zscore'].min():.2f}, {df['LS_zscore'].max():.2f}]")
        print(f"  SS_zscore range: [{df['SS_zscore'].min():.2f}, {df['SS_zscore'].max():.2f}]")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features in the correct order for model input.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with all required columns
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix ready for model inference
        """
        # Model expects features in this exact order
        feature_order = ['Age', 'PLT', 'CPT_Score', 'Viral Etiology', 'Sex', 
                        'LS_zscore', 'SS_zscore']
        
        X = df[feature_order].copy()
        
        # Check for missing values
        missing_counts = X.isnull().sum()
        if missing_counts.any():
            print("\n⚠ Warning: Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  - {col}: {count} missing values")
            print("\nNote: Random Forest can handle missing values using surrogate splits.")
            print("However, performance may be degraded. Consider imputing missing values.\n")
        
        return X
    
    def predict(self, X):
        """
        Run model inference to predict CSPH probability and class.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix prepared for inference
            
        Returns:
        --------
        tuple
            (y_pred_class, y_pred_proba)
            - y_pred_class: Predicted class (0 or 1)
            - y_pred_proba: Predicted probability of CSPH (class 1)
        """
        try:
            # Get predictions
            y_pred_proba = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (CSPH)
            y_pred_class = self.model.predict(X)
            
            print(f"✓ Inference completed successfully")
            print(f"  Mean predicted probability: {y_pred_proba.mean():.3f}")
            print(f"  CSPH cases predicted: {y_pred_class.sum()} / {len(y_pred_class)}")
            
            return y_pred_class, y_pred_proba
            
        except Exception as e:
            print(f"✗ Error during inference: {e}")
            sys.exit(1)
    
    def interpret_score(self, y_pred_proba):
        """
        Interpret ELM Score according to published thresholds.
        
        Parameters:
        -----------
        y_pred_proba : array-like
            Predicted probabilities
            
        Returns:
        --------
        pd.Series
            Risk category for each patient
        """
        categories = []
        for prob in y_pred_proba:
            if prob <= 0.45:
                categories.append('Rule-Out (Low Risk)')
            elif prob >= 0.60:
                categories.append('Rule-In (High Risk)')
            else:
                categories.append('Gray Zone')
        
        return pd.Series(categories, name='Risk_Category')
    
    def run_inference(self, input_file, output_file=None):
        """
        Complete inference pipeline from input file to output predictions.
        
        Parameters:
        -----------
        input_file : str or Path
            Path to input CSV or Excel file
        output_file : str or Path, optional
            Path to save output CSV file. If None, creates output in same directory
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with original data plus predictions
        """
        print("\n" + "="*70)
        print(" ELM Score Inference Pipeline")
        print("="*70 + "\n")
        
        # Step 1: Load data
        print("[1/6] Loading data...")
        df = self.load_data(input_file)
        
        # Step 2: Validate columns
        print("\n[2/6] Validating columns...")
        self.validate_columns(df)
        
        # Step 3: Validate manufacturers
        print("\n[3/6] Validating manufacturers...")
        df = self.validate_manufacturer(df)
        
        if len(df) == 0:
            print("\n✗ Error: No valid rows remaining after manufacturer validation")
            sys.exit(1)
        
        # Step 4: Normalize stiffness values
        print("\n[4/6] Normalizing stiffness measurements...")
        df = self.normalize_stiffness(df)
        
        # Step 5: Prepare features
        print("\n[5/6] Preparing features for inference...")
        X = self.prepare_features(df)
        
        # Step 6: Run inference
        print("\n[6/6] Running model inference...")
        y_pred_class, y_pred_proba = self.predict(X)
        
        # Add predictions to dataframe
        df['ELM_Score'] = y_pred_proba
        df['CSPH_Predicted'] = y_pred_class
        df['Risk_Category'] = self.interpret_score(y_pred_proba)
        
        # Summary
        print("\n" + "="*70)
        print(" Inference Results Summary")
        print("="*70)
        print(f"\nTotal predictions: {len(df)}")
        print(f"\nRisk stratification:")
        risk_counts = df['Risk_Category'].value_counts()
        for category, count in risk_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category:25s}: {count:4d} ({percentage:5.1f}%)")
        
        # Save output
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_predictions.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        print("="*70 + "\n")
        
        return df


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='ELM Score Inference - Predict Clinically Significant Portal Hypertension (CSPH)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python elm_score_inference.py --input patient_data.csv --model rf_model.joblib --params z_score_params.json
  python elm_score_inference.py -i data.xlsx -m model.joblib -p params.json -o results.csv

Required input columns:
  - Age: Patient age in years
  - Sex: 0 = Female, 1 = Male  
  - PLT: Platelet count (×10⁹/L)
  - CPT_Score: Child-Pugh-Turcotte Score (5, 6, or 7)
  - Viral Etiology: 0 = No viral etiology, 1 = Yes (HCV or HBV)
  - Alcohol Etiology: 0 = No alcohol associated liver disease, 1 = Yes
  - Metabolic Etiology: 0 = No metabolic dysfunction-associated steatotic liver disease, 1 = Yes
  - Autoimmune Etiology: 0 = No autoimmune etiology, 1 = Yes
  - Other/mixed Etiology: 0 = No other/mixed etiology, 1 = Yes
  - LS: Liver Stiffness Measurement in kPa
  - SS: Spleen Stiffness Measurement in kPa
  - Elastography Modality: Elastography device ('VCTE-50Hz', 'VCTE-100Hz', 'P-SWE', or '2D-SWE')

Output columns added:
  - LS_zscore: Z-score normalized liver stiffness
  - SS_zscore: Z-score normalized spleen stiffness
  - ELM_Score: Predicted probability of CSPH (0-1)
  - CSPH_Predicted: Binary prediction (0 = No CSPH, 1 = CSPH)
  - Risk_Category: 'Rule-Out (Low Risk)', 'Gray Zone', or 'Rule-In (High Risk)'

Interpretation:
  - ELM Score ≤ 0.45: Rule-Out (Low Risk of CSPH)
  - ELM Score ≥ 0.60: Rule-In (High Risk of CSPH)  
  - ELM Score 0.45-0.60: Gray Zone (Indeterminate)
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input CSV or Excel file with patient data'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='rf_model.joblib', 
        help='Path to trained Random Forest model (.joblib file)'
    )
    
    parser.add_argument(
        '-p', '--params',
        default='z_score_params.json',
        help='Path to z-score normalization parameters (.json file)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Path to output CSV file (default: input_filename_predictions.csv)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = ELMScoreInference(
        z_score_params_path=args.params,
        model_path=args.model
    )
    
    # Run inference
    results = engine.run_inference(
        input_file=args.input,
        output_file=args.output
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())