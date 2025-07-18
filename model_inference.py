import pandas as pd
import xgboost as xgb
import joblib
import os
import argparse

def get_premium_quote(new_applicant_data, model_bundle_file='underwriting_bundle.joblib'):
    """
    Loads a trained underwriting model bundle and provides a premium quote for new applicants.
    """
    print("--- Starting Underwriting Quoting Engine ---")
    
    # --- Load the single model bundle file ---
    if not os.path.exists(model_bundle_file):
        print(f"Error: Model bundle file not found at '{model_bundle_file}'")
        print("Please run the training script first.")
        return None
        
    model_bundle = joblib.load(model_bundle_file)
    print(f"Model bundle '{model_bundle_file}' loaded successfully.")
    
    # --- Unpack the model and columns from the bundle ---
    model = model_bundle['model']
    model_columns = model_bundle['columns']
    
    # --- Preprocess the new data ---
    # The model was not trained on customer_id or customer_name, so we can pass the
    # full dataframe. The reindex step will automatically select only the columns
    # the model needs and was trained on.
    new_df_processed = pd.get_dummies(new_applicant_data, columns=['vehicle_type'])
    new_df_aligned = new_df_processed.reindex(columns=model_columns, fill_value=0)
    
    print("\nGenerating quotes for new applicants...")
    predictions = model.predict(new_df_aligned)
    
    return predictions


if __name__ == "__main__":
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Get premium quotes for new applicants from a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file containing applicant data.")
    args = parser.parse_args()

    inference_file = args.input_file
    
    if not os.path.exists(inference_file):
        print(f"Error: Input file not found at '{inference_file}'")
        print("Please ensure the file exists and the path is correct.")
    else:
        # 1. Load the applicant data from the CSV file
        print(f"\nLoading new applicants from '{inference_file}'...")
        new_applicants_df = pd.read_csv(inference_file)
        
        # 2. Get Predictions
        predicted_premiums = get_premium_quote(new_applicants_df)
        
        # 3. Display Results
        if predicted_premiums is not None:
            print("\n--- Instant Premium Quotation Results ---")
            # Use the original dataframe which includes names and IDs for the final report
            results_df = new_applicants_df.copy()
            results_df['predicted_annual_premium'] = [f"${premium:,.2f}" for premium in predicted_premiums]
            
            # Select and reorder columns for a cleaner final report
            report_columns = [
                'customer_id', 
                'customer_name', 
                'age', 
                'vehicle_type', 
                'vehicle_value', 
                'predicted_annual_premium'
            ]
            print(results_df[report_columns].to_string())
