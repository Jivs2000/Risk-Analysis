#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st


# In[3]:


# Load the dataset
data = pd.read_csv("credit_rating.csv")


# In[5]:


# Show how many missing values are present before cleaning
print("Missing values before cleaning:\n", data.isnull().sum())


# In[7]:


# Drop rows with any missing values
data_cleaned = data.dropna()


# In[9]:


# Show shape before and after
print("\nOriginal shape:", data.shape)
print("Cleaned shape:", data_cleaned.shape)


# In[11]:


# Features and target
X = data.drop('default', axis=1)
y = data['default']


# In[13]:


# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[15]:


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[17]:


# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[19]:


# Step 6: Predict on test set
y_pred = model.predict(X_test)


# In[21]:


# Step 7: Evaluate the model
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[23]:


print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))


# In[29]:


# Step 8 (Optional): Predict on a new customer
new_customer = pd.DataFrame([[40, 2, 10, 5, 50000, 10.5, 2.5, 3.0]],
                            columns=X.columns)
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)
print("\n✅ Predicted default risk (1 = Default, 0 = No Default):", prediction[0])


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


# Get feature importances from the model
importances = model.feature_importances_
feature_names = X.columns


# In[35]:


# Create a DataFrame for plotting
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)


# In[41]:


# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importance in Credit Default Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# In[47]:


# Sidebar: Get input from user
st.sidebar.header("Enter Customer Information")


# In[59]:


age = st.sidebar.slider("Age", 18, 75, 30)
employment_years = st.sidebar.slider("Years of Employment", 0, 40, 5)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", value=50000)
dependents = st.sidebar.slider("Number of Dependents", 0, 10, 1)
annual_income = st.sidebar.number_input("Annual Income (₹)", value=600000)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 40.0, 10.5)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
delinquencies = st.sidebar.slider("Past Delinquencies", 0, 10, 1)


# In[57]:


# Predict on user input
if st.sidebar.button("🔍 Predict Credit Risk"):
    customer_data = pd.DataFrame([[
        age, employment_years, loan_amount, dependents,
        annual_income, interest_rate, credit_score, delinquencies
    ]], columns=X.columns)

    customer_scaled = scaler.transform(customer_data)
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]

    # Determine risk level
    if probability < 0.3:
        risk = "🟢 Low Risk"
    elif probability < 0.7:
        risk = "🟠 Medium Risk"
    else:
        risk = "🔴 High Risk"

    # Show results
    st.sidebar.markdown(f"### 🧠 Prediction: `{risk}`")
    st.sidebar.progress(probability)
    st.sidebar.caption(f"Estimated Default Probability: **{probability:.2%}**")


# In[ ]:




