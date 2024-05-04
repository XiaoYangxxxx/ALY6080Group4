import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset
data = pd.read_csv('df_cleaned.csv')

# Filter for diabetics if necessary and encode categorical variables
data = data[data['Diabetic'] == 'Yes']  # Adjust based on your dataset's encoding
data['PhysicalActivity'] = data['PhysicalActivity'].apply(lambda x: 1 if x == 'Active' else 0)

# Define the features and target
X = data[['PhysicalActivity']]
y = data['HeartDisease']        

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))




###########random forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('/Users/akashkatragadd/Desktop/df_cleaned (1).csv')

# Filter for diabetics and encode categorical variables
data = data[data['Diabetic'] == 'Yes']  # Adjust based on your dataset's encoding
data['PhysicalActivity'] = data['PhysicalActivity'].apply(lambda x: 1 if x == 'Active' else 0)

# Define features and target variable
X = data[['PhysicalActivity']]  # Could add more features here
y = data['HeartDisease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print results
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Optionally, print feature importances
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)