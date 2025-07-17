import joblib
import numpy as np
import pandas as pd

# Step 1: Load the saved model
model = joblib.load("models/student_model.pkl")

# Step 2: Prediction function
def predict_performance(study_hours, attendance, assignments_done,
                        internet_access, parent_support, health_issues):
    
    # Input must be in 2D array format
    input_data = pd.DataFrame([[
    study_hours, attendance, assignments_done,
    internet_access, parent_support, health_issues
]], columns=['study_hours', 'attendance', 'assignments_done',
             'internet_access', 'parent_support', 'health_issues'])
    
    # Step 3: Predict using model
    result = model.predict(input_data)[0]
    
    return "Pass ✅" if result == 1 else "Fail ❌"

# Step 4: Test it
if __name__ == "__main__":
    output = predict_performance(5, 75, 1, 1, 1, 0)
    print("Prediction:", output)
