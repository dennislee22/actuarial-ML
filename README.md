# Actuarial Underwriting with XGBoost-Dask

<img width="295" height="408" alt="image" src="https://github.com/user-attachments/assets/102080c6-3fc2-4d6a-8fdc-c3425c295fe4" />

For decades, insurance underwriting has been a cornerstone of risk management - a meticulous, human-driven process where actuaries and underwriters analyze applicant data to assess risk and determine premiums. While effective, this traditional approach is often slow and relies on generalized models that can struggle to capture the nuances of individual risk.
Machine Learning (ML) is fundamentally reshaping this landscape. By leveraging sophisticated algorithms, insurers can now automate and expedite the underwriting process, moving from manual, days-long assessments to instant, data-driven quotations. In this post, I'll walk through a practical example of how an ML model, specifically XGBoost, can be trained to create a powerful and precise quoting engine.

The goal is to build a system that can take an applicant's profile and instantly generate a fair and accurate annual premium quote. Our workflow is split into two key phases: training and inference.

## Learning from the Past: Training the Model
First, the model needs to learn. We start with a large historical dataset containing tens of thousands of applicant profiles, each with a known "correct" premium quote. The key features in this dataset include:

  - Driver Demographics: Age, driving experience.
  - Vehicle Information: Vehicle type (Sedan, SUV, Sports Car), vehicle value.
  - Usage Patterns: Annual mileage.
  - Risk History: Number of past claims and traffic violations.

The algorithm of choice for this task is XGBoost. It's incredibly fast, highly accurate, and excels at uncovering complex, non-linear relationships in tabular data; like how the risk of a young driver in a sports car increases exponentially, not linearly.
The model then trains on this processed data, iteratively building hundreds of decision trees to learn the intricate patterns that connect an applicant's profile to their premium.
Achieving this automated system involves a clear, four-step process. Here’s how it works in practice:

## Step 1: Generate Synthetic Data
The [synthetic data creation](synthetic_data.py) script generates a large, realistic training dataset (underwriting_data.csv). This file contains thousands of fictional applicant profiles, where each column represents a key risk factor:

  - age & driving_experience: Younger, less experienced drivers typically represent higher risk.
  - vehicle_type & vehicle_value: High-performance or high-value cars are more expensive to insure.
  - annual_mileage: The more you drive, the higher the exposure to potential accidents.
  - past_claims_count & traffic_violations_count: Past behavior is a strong predictor of future risk.
  - annual_premium_quote: The final premium the model will learn to predict.

```
$ python synthetic_data.py
Starting underwriting data generation for 50000000 records...
  ...Generated and saved records 1 to 5000000
  ...Generated and saved records 5000001 to 10000000
  ...Generated and saved records 10000001 to 15000000
  ...Generated and saved records 15000001 to 20000000
  ...Generated and saved records 20000001 to 25000000
  ...Generated and saved records 25000001 to 30000000
  ...Generated and saved records 30000001 to 35000000
  ...Generated and saved records 35000001 to 40000000
  ...Generated and saved records 40000001 to 45000000
  ...Generated and saved records 45000001 to 50000000
Synthetic underwriting data generation complete. Data saved to 'underwriting_data.csv'.
```

## Step 2: Train the XGBoost Model in Distributed Fashion
The [model training script](dask-xgboost-actuarial.ipynb) takes the synthetic data from Step 1 and uses it to train our model. It performs the crucial `One-Hot Encoding` step to convert text to numbers, then feeds the data to the XGBoost algorithm. Models like XGBoost can't work directly with text like "Sedan" or "SUV". They need numbers. `One-Hot Encoding` solves this by creating new columns for each unique category.

For example, if your vehicle_type column has three options ('Sedan', 'SUV', 'Sports Car'), the line of code `dask.dataframe.get_dummies(ddf, columns=['vehicle_type']` will transform the output into numerical value. After learning the patterns, the script saves the complete, trained model into a single, convenient file (underwriting_bundle.joblib) for later use.

### Handling Large-Scale Dataset with Dask

As datasets grow into the tens of GB, they can no longer fit into the limited RAM of the node training the model. To solve this, I use Dask. Dask is a parallel computing library that allows our script to read and process the data in manageable chunks/partitions. By using `dask-xgboost`, model can be trained on the entire dataset without ever needing to load it all into memory at once, making it possible to work with massive amounts of data on a single machine or a cluster.

![actuarial-dask](https://github.com/user-attachments/assets/c6a0c973-e4da-4442-8728-b6423fa7028a)

## Step 3: Create a New Applicant List
To simulate a real-world scenario, this [script](new_customer.py) creates a small CSV file (new_applicants.csv) containing 10 new, unseen customer profiles. This represents a list of potential customers who have just applied for an insurance quote online.

## Step 4: Infer Premiums with the Trained Model
The [inference script](model_inference.py) acts as the quoting engine. It loads the saved model from Step 2 and the new applicant list from Step 3. For each applicant, it processes their data to match the format the model expects and instantly generates a predicted annual premium. This step demonstrates how the system can provide instant quotes at scale.

```
$ python model_inference.py new_applicants.csv 

Loading new applicants from 'new_applicants.csv'...
--- Starting Underwriting Quoting Engine ---
Model bundle 'underwriting_bundle.joblib' loaded successfully.

Generating quotes for new applicants...

--- Instant Premium Quotation Results ---
   customer_id     customer_name  age  vehicle_type  vehicle_value predicted_annual_premium
0         1001        John Smith   50    Sports Car          68207                $6,963.46
1         1002      Maria Garcia   55  Luxury Sedan          30351                $2,527.41
2         1003      David Miller   52    Sports Car          65675                $6,565.02
3         1004       Sarah Jones   59         Sedan          46747                $3,221.64
4         1005      Kenji Tanaka   42           SUV          19851                $1,204.13
5         1006       Emily White   31         Sedan          30007                $1,887.10
6         1007  Carlos Rodriguez   31     Hatchback          74982                $4,833.50
7         1008   Fatima Al-Fassi   49         Sedan          58116                $3,800.54
8         1009      Chloe Dubois   28           SUV          61512                $3,727.23
9         1010    Liam O'Connell   59     Hatchback          71157                $3,391.00
```

### The Model's Report Card: Interpreting the Results
After training, we evaluate the model on a "test set", data it has never seen before. The trained model produced the following results:
```
--- Starting Dask-based Underwriting Model Training Process ---
Dask client created. Dashboard at: <Client: 'tcp://10.42.1.124:8786' processes=5 threads=160, memory=36.78 GiB>
Loading data from 'underwriting_data.csv' with Dask...
Performing Dask feature engineering (One-Hot Encoding)...
Data split into training and testing sets.

Training Dask-XGBoost Regressor model for premium quotation...
Model training complete.

Model and column info bundled and saved to 'underwriting_bundle.joblib'.

--- Evaluating Model Performance on Test Set ---
R-squared (R²): 0.9963
Mean Absolute Error (MAE): $126.23
````

### R-squared (R²): A Measure of Understanding

R-squared measures how much of the change in the premium price is explained by the applicant's details (age, vehicle type, etc.). It's a score between 0 and 1. An R² score of 0.9963 is exceptionally high. It means that 99.63% of the variation in the premium quotes can be explained by the features your model used. In other words, the model has done an excellent job of finding the underlying patterns in the data.
 
### Mean Absolute Error (MAE): A Measure of Precision

The MAE gives us a concrete, real-world measure of the model's average error. An MAE of $127.16 means that, on average, the premium quote generated by our model is only about $126 different from the target premium in our dataset. The MAE gives you a concrete dollar amount for its average error, letting you know that any given quote from the model will likely be within about $126 of the ideal price.

### Business Impact
- The most obvious benefit is speed. An ML model can generate a quote in milliseconds, allowing an insurer to serve thousands of customers online in the time it would take a human underwriter to process a single application.
- The model assesses each applicant on their unique combination of risk factors, moving beyond broad risk pools to provide quotes that are fairer and more accurately reflect individual risk.
- ML automates the repetitive, calculation-intensive work, freeing up highly skilled actuaries to focus on more strategic tasks.

