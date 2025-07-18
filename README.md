# Actuarial Underwriting with ML

For decades, insurance underwriting has been a cornerstone of risk management - a meticulous, human-driven process where actuaries and underwriters analyze applicant data to assess risk and determine premiums. While effective, this traditional approach is often slow and relies on generalized models that can struggle to capture the nuances of individual risk.
Machine Learning (ML) is fundamentally reshaping this landscape. By leveraging sophisticated algorithms, insurers can now automate and expedite the underwriting process, moving from manual, days-long assessments to instant, data-driven quotations. In this post, I'll walk through a practical example of how an ML model, specifically XGBoost, can be trained to create a powerful and precise quoting engine.


The goal is to build a system that can take an applicant's profile and instantly generate a fair and accurate annual premium quote. Our workflow is split into two key phases: training and inference.

## Learning from the Past: Training the Model
First, the model needs to learn. We start with a large historical dataset containing tens of thousands of applicant profiles, each with a known "correct" premium quote. The key features in this dataset include:

  - Driver Demographics: Age, driving experience.
  - Vehicle Information: Vehicle type (Sedan, SUV, Sports Car), vehicle value.
  - Usage Patterns: Annual mileage.
  - Risk History: Number of past claims and traffic violations.

The algorithm of choice for this task is XGBoost. It's incredibly fast, highly accurate, and excels at uncovering complex, non-linear relationships in tabular data—like how the risk of a young driver in a sports car increases exponentially, not linearly.

Before training, we must address a fundamental challenge: models like XGBoost are mathematical and cannot understand text. This is where One-Hot Encoding comes in. We convert a categorical column like vehicle_type into multiple numerical columns (vehicle_type_Sedan, vehicle_type_SUV, etc.), allowing the model to process the information.

The model then trains on this processed data, iteratively building hundreds of decision trees to learn the intricate patterns that connect an applicant's profile to their premium.

A Practical Walkthrough: From Code to Quote
Achieving this automated system involves a clear, four-step process. Here’s how it works in practice:

## Step 1: Generate Synthetic Data
The [synthetic data creation](synthetic_data.py) script generates a large, realistic training dataset (underwriting_data.csv). This file contains thousands of fictional applicant profiles, where each column represents a key risk factor:

  - age & driving_experience: Younger, less experienced drivers typically represent higher risk.
  - vehicle_type & vehicle_value: High-performance or high-value cars are more expensive to insure.
  - annual_mileage: The more you drive, the higher the exposure to potential accidents.
  - past_claims_count & traffic_violations_count: Past behavior is a strong predictor of future risk.
  - annual_premium_quote: The variable—the final premium the model will learn to predict.

```

```

## Step 2: Train the XGBoost Model
The second script takes the synthetic data from Step 1 and uses it to train our model. It performs the crucial `One-Hot Encoding` step to convert text to numbers, then feeds the data to the XGBoost algorithm. After learning the patterns, the script saves the complete, trained model into a single, convenient file (underwriting_bundle.joblib) for later use.

## Step 3: Create a New Applicant List
To simulate a real-world scenario, this [script](new_customer.py) creates a small CSV file (new_applicants.csv) containing 10 new, unseen customer profiles. This represents a list of potential customers who have just applied for an insurance quote online.

## Step 4: Infer Premiums with the Trained Model
The [inference script](model_inference.py) acts as the quoting engine. It loads the saved model from Step 2 and the new applicant list from Step 3. For each applicant, it processes their data to match the format the model expects and instantly generates a predicted annual premium. This step demonstrates how the system can provide instant quotes at scale.

## The Model's Report Card: Interpreting the Results
After training, we evaluate the model on a "test set"—data it has never seen before. This tells us how well it will perform in the real world. Our model produced the following results:

- R-squared (R²): 0.9962
- Mean Absolute Error (MAE): $127.16

These numbers aren't just abstract metrics; they are a direct measure of the model's competence.

### R-squared (R²): A Measure of Understanding

An R² score of 0.9962 is exceptionally high. It means that 99.62% of the variability in the premium prices is successfully explained by the applicant data we provided. This tells us the model isn't just guessing. It has learned the underlying pricing logic from the data with remarkable accuracy. It understands which factors are most important and how they interact to determine risk.

### Mean Absolute Error (MAE): A Measure of Precision

The MAE gives us a concrete, real-world measure of the model's average error. An MAE of $127.16 means that, on average, the premium quote generated by our model is only about $127 different from the target premium in our dataset. For annual premiums that can easily run into the thousands, this level of precision is more than acceptable for an automated, instant quoting system.

### Business Impact
- The most obvious benefit is speed. An ML model can generate a quote in milliseconds, allowing an insurer to serve thousands of customers online in the time it would take a human underwriter to process a single application.
- The model assesses each applicant on their unique combination of risk factors, moving beyond broad risk pools to provide quotes that are fairer and more accurately reflect individual risk.
- ML automates the repetitive, calculation-intensive work, freeing up highly skilled actuaries to focus on more strategic tasks.

