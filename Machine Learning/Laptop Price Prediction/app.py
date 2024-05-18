import numpy as np
import pickle
import streamlit as st

# loading the saved model
XGB_model = pickle.load(open("C:/Users/Josh Peter/Desktop/last deploy/best_model.pkl", 'rb'))
df = pickle.load(open("C:/Users/Josh Peter/Desktop/last deploy/data1.pkl", 'rb'))

# Mapping dictionaries
Company_map = {'Acer': 0, 'Apple': 1, 'Asus': 2, 'Chuwi': 3, 'Dell': 4, 'Fujitsu': 5, 'Google': 6, 'HP': 7, 'Huawei': 8, 'LG': 9, 'Lenovo': 10, 'MSI': 11, 'Mediacom': 12, 'Microsoft': 13, 'Razer': 14, 'Samsung': 15, 'Toshiba': 16, 'Vero': 17, 'Xiaomi': 18}

TypeName_map = {'2 in 1 Convertible': 0, 'Gaming': 1, 'Netbook': 2, 'Notebook': 3, 'Ultrabook': 4, 'Workstation': 5}

Cpu_brand_map = {'AMD Processor': 0, 'Intel Core i3': 1, 'Intel Core i5': 2, 'Intel Core i7': 3, 'Other Intel Processor': 4}

gpu_brand_map = {'AMD': 0, 'ARM': 1, 'Intel': 2, 'Nvidia': 3}

Operte_sys_map = {'Android': 0, 'Chrome OS': 1, 'Linux': 2, 'Mac': 3, 'No OS': 4, 'Windows': 5}

def Laptop_price_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = XGB_model.predict(input_data_reshaped)
    j = np.exp(prediction[0])
    return j

def main():
    # giving a title
    st.title('Laptop Price Prediction')

    # getting the input data from the user
    Company = st.selectbox('Brand', list(Company_map.keys()))
    TypeName = st.selectbox('Type Name', list(TypeName_map.keys()))
    RAM = st.selectbox('RAM', sorted(df['RAM'].unique()))
    SSD = st.selectbox('SSD', sorted(df['SSD'].unique()))
    HDD = st.selectbox('HDD', sorted(df['HDD'].unique()))
    Hybrid = st.selectbox('Hybrid', sorted(df['Hybrid'].unique()))
    Flash_storage = st.selectbox('Flash storage', sorted(df['Flash_storage'].unique()))
    Inches = st.slider('Inches', min_value=10.0, max_value=20.0, step=0.1)
    Screen_reso = st.selectbox('Screen Resolution', ['1366x768', '1440x900', '1600x900', '1920x1080', '1920x1200',
                                                     '2160x1440', '2256x1504', '2304x1440', '2400x1600', '2560x1440',
                                                     '2560x1600', '2736x1824', '2880x1800', '3200x1800', '3840x2160'])
    Touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
    Ips = st.selectbox('Ips', ['Yes', 'No'])
    Cpu_brand = st.selectbox('Cpu brand', list(Cpu_brand_map.keys()))
    gpu_brand = st.selectbox('Gpu brand', list(gpu_brand_map.keys()))

    # Adjust OS options based on selected company
    if Company == 'Apple':
        os_options = ['Mac', 'No OS']
    else:
        os_options = list(Operte_sys_map.keys())
    
    Operte_sys = st.selectbox('OS', os_options)
    
    Weight = st.slider('Weight', min_value=0.68, max_value=5.0, step=0.01)

    # code for Prediction
    Price = ''

    if st.button('Price Result'):
        ppi = None
        # Convert encoded label back to category using if-else statements
        Company_encoded = Company_map.get(Company, 'Xiaomi')
        TypeName_encoded = TypeName_map.get(TypeName, 'Workstation')
        Touchscreen_encoded = 1 if Touchscreen == 'Yes' else 0
        Ips_encoded = 1 if Ips == 'Yes' else 0
        X_res = int(Screen_reso.split('x')[0])
        Y_res = int(Screen_reso.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2))
        ppi = round(((ppi ** 0.5) / Inches), 5)

        Cpu_brand_encoded = Cpu_brand_map.get(Cpu_brand, 'Other Intel Processor')
        gpu_brand_encoded = gpu_brand_map.get(gpu_brand, 'Nvidia')
        Operte_sys_encoded = Operte_sys_map.get(Operte_sys, 'Windows')

        Price = Laptop_price_prediction([Company_encoded, TypeName_encoded, Weight, Touchscreen_encoded, Ips_encoded, ppi,
                                         Cpu_brand_encoded, RAM, SSD, HDD,
                                         Hybrid, Flash_storage, gpu_brand_encoded, Operte_sys_encoded])
        st.write("The Laptop price is:")
        st.success(Price)
        st.write('PPI', ppi)
        print(Price)
    
    # Display reference prices for selected company and type name, sorted by price
    st.write(f"Other Available models in {Company} ({TypeName}):")
    # here the "PRICE" is in Capitals as the columns name is also in Captials
    if 'PRICE' in df.columns:
        reference_prices = df[(df['Company'] == Company) & (df['TypeName'] == TypeName)][['Company', 'TypeName', 'RAM', 'SSD', 'Cpu_brand', 'Operte_sys', 'gpu_brand', 'PRICE']].sort_values(by='PRICE')
        st.dataframe(reference_prices)

if __name__ == '__main__':
    main()