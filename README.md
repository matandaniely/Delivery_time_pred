# Course Project – ML Ops - Delivery Time
Predict the time taken by the delivery person to deliver the food from the restaurant to the delivery location



### Problem Statement 
Predict the time taken by the delivery person to deliver the food from the restaurant to the delivery location. With the help of:
•	Age of the delivery person (Delivery_person_age)
•	Previous rating (Delivery_person_Rating)
•	Distance between restaurant and delivery location (Restaurant/Delivery_location_longtitude/latitude)

Dataset: Delivery Time (Kaggle)
Features: 
ID	Delivery_person_ID	Delivery_person_Age	Delivery_person_Ratings	Restaurant_latitude	Restaurant_longitude	Delivery_location_latitude	Delivery_location_longitude	Type_of_order	Type_of_vehicle	Time_taken(min)

Project Type: Regression, and the target = Time_taken(min)
Feature engineering steps



Graphs comparing the performance

# trigger CD
Course Project – ML Ops: Delivery Time Prediction
Problem Statement
The goal of this project is to predict the time taken (in minutes) for a delivery person to deliver food from the restaurant to the customer.
This is a regression problem, where the target variable is:
 Time_taken(min)
Key features used for prediction:
•	Delivery_person_Age – Age of the delivery person
•	Delivery_person_Ratings – Previous rating of the delivery person
•	Geolocation data:
o	Restaurant_latitude, Restaurant_longitude
o	Delivery_location_latitude, Delivery_location_longitude
•	Type_of_order – Type of food order
•	Type_of_vehicle – Type of vehicle used
________________________________________
Our Approach
1.	Data Preprocessing & Feature Engineering
o	Cleaned missing and invalid entries
o	Calculated distance between restaurant and delivery location using geopy
o	Encoded categorical features (Type_of_order, Type_of_vehicle)
2.	Model Training
o	Trained a Linear Regression model
o	Used MLflow to log parameters, metrics, and artifacts (models)
3.	CI/CD Integration
o	Implemented CI (Continuous Integration) with GitHub Actions to:
 	Automatically test and lint code upon each push
o	Implemented CD (Continuous Deployment) to:
	Automatically re-run the pipeline and upload new artifacts
	Triggered only if the CI workflow completes successfully
________________________________________
Results & Tracking
•	Evaluated model performance using metrics like R² Score
•	Plotted residual errors and prediction vs actual values
•	Logged all experiments in MLflow UI
________________________________________
🔗 MLflow Tracking
Our experiments are tracked using MLflow Tracking Server, hosted at:
cpp

http://127.0.0.1:8081
Each experiment contains:
•	Model input parameters
•	Performance metrics
•	Downloadable trained model (artifact)
________________________________________
Status
•	CI Pipeline: Working
•	CD Pipeline: Running successfully after linking MLflow
•	All experiments reproducible and tracked

