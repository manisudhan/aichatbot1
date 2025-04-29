import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('data2.csv')

# Data Preprocessing
data['Case Description'] = data['Case Description'].fillna('')  # Handle missing values

# Step 3: Convert Text to Numerical Features
vectorizer = TfidfVectorizer(max_features=500)  # Use TF-IDF for text vectorization
X = vectorizer.fit_transform(data['Case Description']).toarray()  # Feature matrix from case descriptions
y = data['Weightage Score']  # Target variable

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
