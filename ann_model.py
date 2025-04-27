import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from tensorflow.keras.models import load_model  # Tambahan untuk menyimpan model

# Load dataset
df = pd.read_csv("house_price.csv")

# Kolom yang dipakai
columns = [
    'building_size_m2', 'land_size_m2', 'bedrooms', 'bathrooms',
    'floors', 'district', 'property_condition', 'furnishing',
    'garages', 'price_in_rp'
]
df = df[columns].dropna()

# Tambahkan kolom 'year'
current_year = datetime.now().year
df['year'] = current_year

# Fitur dan target
X = df.drop('price_in_rp', axis=1)
y = df['price_in_rp']

# Kolom numerik dan kategorik
numerical = ['building_size_m2', 'land_size_m2', 'bedrooms', 'bathrooms', 'floors', 'garages', 'year']
categorical = ['district', 'property_condition', 'furnishing']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Transformasi data
X_processed = preprocessor.fit_transform(X)

# Simpan preprocessor
os.makedirs("model", exist_ok=True)
joblib.dump(preprocessor, "model/preprocessor.pkl")

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Bangun model ANN regresi
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Latih model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Simpan model
model.save('model/ann_regresi_rumah.h5')
print("Model ANN berhasil dilatih dan disimpan!")

# Evaluasi dan prediksi
y_pred = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"MAE untuk data uji: {mae_test:.2f} Rp")

mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
print(f"MAPE untuk data uji: {mape:.2f}%")

# Visualisasi: Harga Asli vs Prediksi
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Perbandingan Harga Asli vs Prediksi')
plt.xlabel('Harga Rumah Asli (Rp)')
plt.ylabel('Harga Rumah Prediksi (Rp)')
plt.show()

# Visualisasi: Distribusi Error
errors = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribusi Error Prediksi Harga Rumah')
plt.xlabel('Error (Selisih Harga)')
plt.ylabel('Frekuensi')
plt.show()

# Prediksi masa depan
years_to_predict = [current_year + i for i in range(1, 6)]
future_data = {
    'building_size_m2': [300] * 5,
    'land_size_m2': [200] * 5,
    'bedrooms': [3] * 5,
    'bathrooms': [2] * 5,
    'floors': [2] * 5,
    'district': ['A'] * 5,
    'property_condition': ['Good'] * 5,
    'furnishing': ['Furnished'] * 5,
    'garages': [2] * 5,
    'year': years_to_predict
}
future_df = pd.DataFrame(future_data)
future_processed = preprocessor.transform(future_df)
future_predictions = model.predict(future_processed)

plt.figure(figsize=(10, 6))
plt.plot(years_to_predict, future_predictions, marker='o', color='green')
plt.title('Prediksi Harga Rumah untuk Beberapa Tahun ke Depan')
plt.xlabel('Tahun')
plt.ylabel('Harga Rumah Prediksi (Rp)')
plt.show()
