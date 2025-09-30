import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

# Load datasets
cancer_data = pd.read_excel('cancer_data.xlsx')
malignant_data = pd.read_excel('Copy of malignant_tumor_dataset(1).xlsx')

# Preprocessing cancer_data
X = cancer_data.drop('Malignant', axis=1)
y = cancer_data['Malignant']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the Logistic Regression model to predict malignancy
model = LogisticRegression()
model.fit(X_train, y_train)

# Initialize session state variables
if 'malignancy_test_completed' not in st.session_state:
    st.session_state['malignancy_test_completed'] = False
if 'malignancy_prediction' not in st.session_state:
    st.session_state['malignancy_prediction'] = None
if 'operability_test_started' not in st.session_state:
    st.session_state['operability_test_started'] = False
if 'operability_test_completed' not in st.session_state:
    st.session_state['operability_test_completed'] = False

# Streamlit app
st.title("Cancer Prediction and Operability Analysis")

# Input for Malignancy Prediction
st.header("Malignancy Prediction")

# Sliders for user input
lipid_count = st.slider("Lipid count (100-300 mg/dL):", 100, 300, 150)
protein_levels = st.slider("Protein levels (6-8 g/dL):", 6.0, 8.0, 7.0)
platelet_count = st.slider("Platelet count (100-400 x 10^3/μL):", 100, 400, 250)
white_blood_cell_count = st.slider("White blood cell count (4000-11000 cells/μL):", 4000, 11000, 7000)
smoking_history = st.selectbox("Smoking history (1: Yes, 0: No):", [0, 1])
age = st.slider("Age (0-90 years):", 0, 90, 45)
tumor_size = st.slider("Tumor size (0-10 cm):", 0.0, 10.0, 3.0)
hepatitis_diagnosis = st.selectbox("Hepatitis diagnosis (1: Yes, 0: No):", [0, 1])
hormonal_imbalance = st.selectbox("Had hormonal replacement/imbalance (1: Yes, 0: No):", [0, 1])
family_history = st.selectbox("Family history of cancer (1: Yes, 0: No):", [0, 1])
had_cancer = st.selectbox("Had cancer previously (1: Yes, 0: No):", [0, 1])
exposed_to_radiation = st.selectbox("Exposed to radiation (1: Yes, 0: No):", [0, 1])
exposed_to_carcinogens = st.selectbox("Exposed to carcinogens (1: Yes, 0: No):", [0, 1])
genetic_mutation = st.selectbox("Genetic mutation (1: Yes, 0: No):", [0, 1])

# Create a DataFrame from the input
new_patient_data = {
    'Lipid Count': [lipid_count],
    'Protein Levels': [protein_levels],
    'Platelet Count': [platelet_count],
    'White Blood Cell Count': [white_blood_cell_count],
    'Smoking History': [smoking_history],
    'Age': [age],
    'Tumor Size': [tumor_size],
    'Hepatitis Diagnosis': [hepatitis_diagnosis],
    'Had Hormonal Replacement/Imbalance': [hormonal_imbalance],
    'Family History': [family_history],
    'Had Cancer Previously': [had_cancer],
    'Exposed to Radiation': [exposed_to_radiation],
    'Exposed to Carcinogens': [exposed_to_carcinogens],
    'Genetic Mutation': [genetic_mutation]
}
new_patient_df = pd.DataFrame(new_patient_data)
new_patient_scaled = scaler.transform(new_patient_df)

# Predict malignancy
if st.button("Predict Malignancy"):
    malignant_prediction_prob = model.predict_proba(new_patient_scaled)[:, 1]
    st.session_state['malignancy_prediction'] = malignant_prediction_prob[0]
    st.session_state['malignancy_test_completed'] = True

# Display malignancy results with refined threshold
if st.session_state['malignancy_test_completed']:
    malignant_prediction_prob = st.session_state['malignancy_prediction']
    
    # Refined threshold for malignancy
    malignancy_threshold = 0.5  # Example threshold, adjust based on latest data insights
    if malignant_prediction_prob > malignancy_threshold:
        st.write('New Prediction: Malignant')
        st.write(f'Probability of being Malignant: {malignant_prediction_prob:.2f}')
        
        # Scatter plot for malignancy
        patient_combined = cancer_data.copy()
        new_patient_df['Malignant'] = 1
        patient_combined = pd.concat([patient_combined, new_patient_df], ignore_index=True)

        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=patient_combined['Age'], y=patient_combined['Tumor Size'], hue=patient_combined['Malignant'], palette='coolwarm', alpha=0.7, ax=ax1)
        ax1.scatter(new_patient_data['Age'], new_patient_data['Tumor Size'], color='red', marker='X', s=100, label='New Patient')
        plt.xlabel('Age')
        plt.ylabel('Tumor Size')
        plt.title('Malignancy Prediction Scatter Plot')
        plt.legend()
        st.pyplot(fig1)

        # Start operability prediction
        st.session_state['operability_test_started'] = True

    else:
        st.write('New Prediction: Benign')
        st.write(f'Probability of being Malignant: {malignant_prediction_prob:.2f}')
        st.session_state['operability_test_started'] = False

# Operability Prediction Inputs with refined logic
if st.session_state['operability_test_started']:
    st.header("Operability Prediction")

    # Operability Input Fields
    tumor_smoothness = st.slider("Tumor smoothness (0-1 scale):", 0.0, 1.0, 0.5)
    aggressive_index = st.slider("Aggressive index (0-10):", 0.0, 10.0, 5.0)
    chemotherapy = st.selectbox("Received chemotherapy (1: Yes, 0: No):", [0, 1])

    # Button to trigger operability check
    if st.button("Check Operability"):
        # Refined operability logic with conditional statements
        if tumor_size > 3.0:
            if aggressive_index > 6.0:
                is_operable = 0  # Tumor is inoperable
            elif tumor_smoothness < 0.6 and chemotherapy == 0:
                is_operable = 0  # Tumor is inoperable
            else:
                is_operable = 1  # Tumor may be operable
        else:
            if aggressive_index > 8.0:
                is_operable = 0  # Tumor is inoperable
            else:
                is_operable = 1  # Tumor is operable
        new_patient_malignant_data = {
            'Tumor Size': [tumor_size],
            'Tumor Smoothness': [tumor_smoothness],
            'Aggressive Index': [aggressive_index],
            'Chemotherapy': [chemotherapy],
            'is_operable': [is_operable]
        }
        new_patient_malignant_df = pd.DataFrame(new_patient_malignant_data)
        # Scatter plot for operability
        patient_combined_op = malignant_data.copy()
        patient_combined_op = pd.concat([patient_combined_op, new_patient_malignant_df], ignore_index=True)
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=patient_combined_op['Tumor Size'], y=patient_combined_op['Aggressive Index'], hue=patient_combined_op['is_operable'], palette='viridis', alpha=0.7, ax=ax2)
        ax2.scatter(new_patient_malignant_df['Tumor Size'], new_patient_malignant_df['Aggressive Index'], color='orange', marker='X', s=100, label='New Patient')
        plt.xlabel('Tumor Size')
        plt.ylabel('Aggressive Index')
        plt.title('Operability Prediction Scatter Plot')
        plt.legend()
        st.pyplot(fig2)
        if is_operable == 1:
            st.write("The tumor is operable.")
        else:
            st.write("The tumor is inoperable.")
            # Predict the time left using KNeighborsRegressor
            knn = KNeighborsRegressor(n_neighbors=5)
            malignant_features = malignant_data[['Tumor Size', 'Tumor Smoothness', 'Aggressive Index']]
            time_left = malignant_data['time_left']
            knn.fit(malignant_features, time_left)
            time_left_prediction = knn.predict(new_patient_malignant_df[['Tumor Size', 'Tumor Smoothness', 'Aggressive Index']])
            st.write(f"Estimated time left: {time_left_prediction[0]:.2f} months.")