import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

st.title("🏋️ Personal Fitness Tracker")
st.write("Predict the calories burned based on your physical attributes and workout details.")

@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = round(df["Weight"] / ((df["Height"] / 100) ** 2), 2)
    df = pd.get_dummies(df, drop_first=True)
    return df

dataset = load_data()

# Sidebar Input Section
st.sidebar.header("🔍 User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Linear Regression"])
    
    gender = 1 if gender_button == "Male" else 0
    
    features = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender]
    })
    return features, model_choice

df, model_choice = user_input_features()
st.write("### Your Input Parameters")
st.dataframe(df)

# Train-Test Split
X = dataset.drop("Calories", axis=1)
y = dataset["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

@st.cache_resource
def train_model(model_choice):
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=6)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(model_choice)

# Ensure input alignment
df = df.reindex(columns=X_train.columns, fill_value=0)

# Prediction
prediction = model.predict(df)

# Display Prediction
st.write("### 🔥 Estimated Calories Burned:")
st.success(f"{round(prediction[0], 2)} kcal")

# Find Similar Results
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = dataset[(dataset["Calories"] >= calorie_range[0]) & (dataset["Calories"] <= calorie_range[1])]

st.write("### 📊 Similar Results")
if not similar_data.empty:
    fig, ax = plt.subplots()
    sns.histplot(similar_data["Calories"], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution of Similar Calorie Burns")
    ax.set_xlabel("Calories Burned")
    st.pyplot(fig)
    st.write(similar_data.sample(min(5, len(similar_data))))
else:
    st.write("No similar records found.")

# General Statistics
st.write("### 📈 Comparative Analysis")
st.write(f"You are older than **{(dataset['Age'] < df['Age'].values[0]).mean() * 100:.2f}%** of users.")
st.write(f"Your workout duration is longer than **{(dataset['Duration'] < df['Duration'].values[0]).mean() * 100:.2f}%** of users.")
st.write(f"Your heart rate is higher than **{(dataset['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100:.2f}%** of users.")
st.write(f"Your body temperature is higher than **{(dataset['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100:.2f}%** of users.")
