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
laptop_data2 = pd.read_csv("laptops.csv")

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
# possible_display_sizes = ["13.3", "14.0", "15.6", "16.0", "17.3"]
# possible_resolution_widths = ["1366", "1920", "2560", "3840"]
# possible_resolution_heights = ["768", "1080", "1440", "2160"]
# possible_num_cores = list(range(2, 17, 2))
# possible_num_threads = list(range(2, 17, 2))
# possible_ram_memory = list(range(4, 33, 4))

# Sidebar Inputs
st.sidebar.title("Laptop Recommendation System")
st.sidebar.write("Enter your preferences and click below!")

user_input = {}

# Select Brand First
brand_options = sorted(laptop_data2["brand"].unique().tolist())
selected_brand = st.sidebar.selectbox("Select Brand", brand_options)

def get_filtered_options(column_name):
    return sorted(laptop_data2[laptop_data2["brand"] == selected_brand][column_name].dropna().unique().tolist())

selected_processor_brand = st.sidebar.selectbox("Select Processor Brand", get_filtered_options("processor_brand"))

filtered_processor_tiers = sorted(
    laptop_data2[
        (laptop_data2["brand"] == selected_brand) & 
        (laptop_data2["processor_brand"] == selected_processor_brand)
    ]["processor_tier"].dropna().unique().tolist()
)

selected_processor_tier = st.sidebar.selectbox("Select Processor Tier", filtered_processor_tiers)
selected_num_cores = st.sidebar.selectbox("Select Number of Cores", get_filtered_options("num_cores"))
selected_num_threads = st.sidebar.selectbox("Select Number of Threads", get_filtered_options("num_threads"))
selected_ram_memory = st.sidebar.selectbox("Select RAM Memory", get_filtered_options("ram_memory"))
selected_gpu_brand = st.sidebar.selectbox("Select GPU Brand", get_filtered_options("gpu_brand"))
selected_gpu_type = st.sidebar.selectbox("Select GPU Type", get_filtered_options("gpu_type"))
selected_display_size = st.sidebar.selectbox("Select Display Size", get_filtered_options("display_size"))
selected_resolution_width = st.sidebar.selectbox("Select Resolution Width", get_filtered_options("resolution_width"))
selected_resolution_height = st.sidebar.selectbox("Select Resolution Height", get_filtered_options("resolution_height"))
selected_os = st.sidebar.selectbox("Select Operating System", get_filtered_options("OS"))

# Store user input
user_input.update({
    "brand": selected_brand,
    "processor_brand": selected_processor_brand,
    "processor_tier": selected_processor_tier,
    "num_cores": selected_num_cores,
    "num_threads": selected_num_threads,
    "ram_memory": selected_ram_memory,
    "gpu_brand": selected_gpu_brand,
    "gpu_type": selected_gpu_type,
    "display_size": selected_display_size,
    "resolution_width": selected_resolution_width,
    "resolution_height": selected_resolution_height,
    "OS": selected_os
})

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

        # Filter laptops based on predicted category AND price range
        price_lower = usd_price - (500 / conversion_rate)
        price_upper = usd_price + (500 / conversion_rate)

        filtered_laptops = laptop_data.copy()
        
        matching_laptops = filtered_laptops[
            (filtered_laptops['Category'] == classification_prediction) &
            (filtered_laptops['Price'] >= price_lower) &
            (filtered_laptops['Price'] <= price_upper)
        ].copy()
        
        if not matching_laptops.empty:
            if matching_laptops['brand'].dtype != 'O':  # Not object → likely numeric
                matching_laptops['brand'] = label_encoders['brand'].inverse_transform(matching_laptops['brand'].astype(int))
            matching_laptops['Price_MYR'] = matching_laptops['Price'] * conversion_rate
            matching_laptops['Price_MYR'] = matching_laptops['Price_MYR'].apply(lambda x: f"RM{x:.2f}")
            st.subheader("Possible Laptop Models (within ±RM500 of your estimated price):")
            st.write(matching_laptops[['Model', 'brand', 'Price_MYR']])
        else:
            st.write("No matching laptops found within your budget and category.")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
