import pandas as pd
import random

def create_applicants_csv(filename='new_applicants.csv', num_records=10):
    """
    Creates a sample CSV file with a specified number of customers for inference.
    """
    print(f"Creating sample CSV file for inference: '{filename}'")
    names = ['John Smith', 'Maria Garcia', 'David Miller', 'Sarah Jones', 'Kenji Tanaka', 
             'Emily White', 'Carlos Rodriguez', 'Fatima Al-Fassi', 'Chloe Dubois', 'Liam O\'Connell']
    vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Sports Car', 'Luxury Sedan']
    
    data = []
    for i in range(num_records):
        record = {
            'customer_id': 1001 + i,
            'customer_name': names[i % len(names)],
            'age': random.randint(20, 70),
            'driving_experience': random.randint(2, 50),
            'vehicle_type': random.choice(vehicle_types),
            'vehicle_value': random.randint(18000, 95000),
            'annual_mileage': random.randint(8000, 35000),
            'past_claims_count': random.choice([0, 1, 2]),
            'traffic_violations_count': random.choice([0, 1, 2])
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Sample CSV with {num_records} customers created successfully.")


if __name__ == "__main__":
    # Define the output file for the applicant data
    applicants_file = 'new_applicants.csv'
    
    # Create the sample CSV file with 10 customers
    create_applicants_csv(applicants_file, num_records=10)

