import pandas as pd
import numpy as np
import random
import os

def generate_applicant_batch(batch_size=1000):
    """
    Generator function that yields a batch of synthetic applicant profiles.
    """
    vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Sports Car', 'Luxury Sedan']
    employment_statuses = ['Employed', 'Self-Employed', 'Unemployed', 'Student']
    
    data = []
    
    for _ in range(batch_size):
        # --- Applicant and Vehicle Profile ---
        age = np.random.randint(18, 75)
        driving_experience = max(1, age - 17 - np.random.randint(0, 5))
        vehicle_type = random.choice(vehicle_types)
        vehicle_value = round(np.random.uniform(15000, 120000))
        annual_mileage = np.random.randint(5000, 40000)
        past_claims_count = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.05, 0.05])
        traffic_violations_count = np.random.choice([0, 1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.05, 0.05])
        
        # --- Calculate a logical Risk Score ---
        # Start with a base risk score
        risk_score = 0.1
        
        # Age and experience are major factors
        if age < 25:
            risk_score += 0.25
        elif age > 65:
            risk_score += 0.1
        if driving_experience < 5:
            risk_score += (5 - driving_experience) * 0.04
            
        # Vehicle type adds risk
        if vehicle_type == 'Sports Car':
            risk_score += 0.3
        elif vehicle_type == 'Luxury Sedan':
            risk_score += 0.15
            
        # Past behavior is highly predictive
        risk_score += past_claims_count * 0.08
        risk_score += traffic_violations_count * 0.06
        
        # High mileage increases exposure
        risk_score += (annual_mileage / 40000) * 0.1
        
        # Clip score to be between 0 and 1
        risk_score = np.clip(risk_score, 0, 1)
        
        # --- Generate Premium Quote based on Risk and Value ---
        # Base premium is a percentage of the vehicle's value
        base_premium = vehicle_value * 0.03
        
        # The risk score acts as a multiplier on the premium
        # A low-risk person (e.g. 0.1) pays close to base, a high-risk (e.g. 0.9) pays much more
        risk_multiplier = 1 + (risk_score * 3.5)
        
        annual_premium = base_premium * risk_multiplier
        
        # Add some random noise
        annual_premium *= np.random.uniform(0.95, 1.05)

        record = {
            'age': age,
            'driving_experience': driving_experience,
            'vehicle_type': vehicle_type,
            'vehicle_value': vehicle_value,
            'annual_mileage': annual_mileage,
            'past_claims_count': past_claims_count,
            'traffic_violations_count': traffic_violations_count,
            'calculated_risk_score': round(risk_score, 4), # For reference
            'annual_premium_quote': round(annual_premium, 2) # This is our target
        }
        data.append(record)
        
    return pd.DataFrame(data)

def create_underwriting_dataset(total_records=50_000_000, batch_size=5_000_000, output_file='underwriting_data.csv'):
    """
    Creates the full underwriting dataset by generating and saving batches.
    """
    print(f"Starting underwriting data generation for {total_records} records...")
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    for i in range(0, total_records, batch_size):
        batch_df = generate_applicant_batch(batch_size)
        header = not os.path.exists(output_file)
        batch_df.to_csv(output_file, mode='a', header=header, index=False)
        print(f"  ...Generated and saved records {i+1} to {i+batch_size}")
        
    print(f"\nSynthetic underwriting data generation complete. Data saved to '{output_file}'.")


if __name__ == "__main__":
    create_underwriting_dataset()
