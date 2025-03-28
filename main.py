import streamlit as st
import joblib
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Navigation")
model_option = st.sidebar.selectbox("Select Recommendation Approach", [
    "Approach 1: Random Forest",
    "Approach 2: KNN + Linear Regression",
    "Approach 3: Naive Bayes + Decision Tree"
])

laptop_data = pd.read_csv("laptops_updated.csv")

# Load corresponding model file
if model_option == "Approach 1: Random Forest":
    model_data = joblib.load("approach1_rf.joblib")
elif model_option == "Approach 2: KNN + Linear Regression":
    model_data = joblib.load("approach2_knn_lr.joblib")
else:
    model_data = joblib.load("approach3_nb_dt.joblib")

# Extract components
clf = model_data["classification_model"]
reg = model_data["regression_model"]
label_encoders = model_data["label_encoders"]
features = model_data["features"]

# Define possible values
possible_display_sizes = ["13.3", "14.0", "15.6", "16.0", "17.3"]
possible_resolution_widths = ["1366", "1920", "2560", "3840"]
possible_resolution_heights = ["768", "1080", "1440", "2160"]
possible_num_cores = list(range(2, 17, 2))
possible_num_threads = list(range(2, 17, 2))
possible_ram_memory = list(range(4, 33, 4))

# Sidebar Inputs
st.sidebar.title("Laptop Recommendation System")
st.sidebar.write("Enter your preferences and click below!")

user_input = {}
for feature in features:
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", options)
    elif feature == "display_size":
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", possible_display_sizes)
    elif feature == "resolution_width":
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", possible_resolution_widths)
    elif feature == "resolution_height":
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", possible_resolution_heights)
    elif feature == "num_cores":
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", possible_num_cores)
    elif feature == "num_threads":
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", possible_num_threads)
    elif feature == "ram_memory":
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", possible_ram_memory)
    else:
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", list(range(1, 17)))

predict_button = st.sidebar.button("Get Recommendation")

# Main Output
st.title("Laptop Recommendation System")
st.write("Recommendations will appear below based on your selected preferences.")

if predict_button:
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=features, fill_value=0)

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df = input_df.astype(float)

    try:
        classification_prediction = int(clf.predict(input_df)[0])
        regression_prediction = reg.predict(input_df)[0]

        category_labels = {1: "Gaming", 2: "Business", 3: "Budget-Friendly"}
        category_name = category_labels.get(classification_prediction, "Unknown")

        usd_price = regression_prediction
        conversion_rate = 4.435  # USD to MYR
        myr_price = usd_price * conversion_rate
        
        st.subheader("Recommendation Results:")
        st.write(f"Predicted Laptop Category: {category_name}")
        st.write(f"Estimated Laptop Price: RM{myr_price:.2f}")

        matching_laptops = laptop_data[laptop_data['Category'] == classification_prediction]
        if not matching_laptops.empty:
            st.subheader("Possible Laptop Models:")
            st.write(matching_laptops[['Model', 'brand', 'Price']])
        else:
            st.write("No matching laptops found in the dataset.")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
