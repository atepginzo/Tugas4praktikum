from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load preprocessor dan model ANN
preprocessor = joblib.load('model/preprocessor.pkl')
model = load_model('model/ann_regresi_rumah.h5', compile=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        # Load dataset
        df = pd.read_csv('house_price.csv')

        # Kolom yang diperlukan
        required_columns = ['building_size_m2', 'land_size_m2', 'bedrooms', 'bathrooms',
                            'floors', 'district', 'property_condition', 'furnishing',
                            'garages', 'year', 'price_in_rp']

        # Pastikan kolom tersedia
        for col in required_columns:
            if col not in df.columns:
                if col == 'year':
                    df[col] = 2023
                else:
                    df[col] = 0

        df = df[required_columns].dropna()

        column_mapping = {
    'building_size_m2': 'Luas Bangunan',
    'land_size_m2': 'Luas Tanah',
    'bedrooms': 'Kamar Tidur',
    'bathrooms': 'Kamar Mandi',
    'floors': 'Jumlah Lantai',
    'district': 'Lokasi',
    'property_condition': 'Kondisi',
    'furnishing': 'Furnitur',
    'garages': 'Garasi',
    'year': 'Tahun',
    'price_in_rp': 'Harga'
}
        df.rename(columns=column_mapping, inplace=True)

        # Ambil lokasi dari dataset
        lokasi_dataset = sorted(df['Lokasi'].dropna().unique().tolist())

        # Ambil lokasi dari model (preprocessor)
        try:
            encoder = preprocessor.named_transformers_['cat']
            lokasi_model = sorted(list(encoder.categories_[0]))
        except Exception as e:
            print(f"Error getting location categories from model: {str(e)}")
            lokasi_model = []

        # Gabungkan → hanya lokasi yang dikenal model DAN ada di dataset
        lokasi_valid = sorted(list(set(lokasi_dataset) & set(lokasi_model)))

        # Filter dataset
        df = df[df['Lokasi'].isin(lokasi_valid)]

        if df.empty:
            raise ValueError("Dataset kosong setelah filtering lokasi valid.")

        # Siapkan input untuk prediksi
        input_columns = ['Luas Bangunan', 'Luas Tanah', 'Kamar Tidur', 'Kamar Mandi',
                        'Jumlah Lantai', 'Lokasi', 'Kondisi', 'Furnitur', 'Garasi', 'Tahun']
        input_X = df[input_columns].copy()

        # Mapping balik ke nama kolom asli untuk preprocessing
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        input_X_original = input_X.rename(columns=reverse_mapping)

        # Preprocessing dan prediksi
        try:
            input_X_processed = preprocessor.transform(input_X_original)
            predictions = model.predict(input_X_processed)
            df['Prediksi Harga'] = predictions.flatten()
        except Exception as e:
            print(f"Error during preprocessing/prediction: {str(e)}")
            raise

        # Tabel untuk ditampilkan
        display_columns = input_columns + ['Harga', 'Prediksi Harga']
        tabel_data = df[display_columns].copy()

        # Format harga dan prediksi
        for col in ['Harga', 'Prediksi Harga']:
            tabel_data[col] = tabel_data[col].apply(lambda x: '{:,.2f}'.format(x))

        # Untuk form input prediksi manual
        prediction = None
        input_data = None
        error = None

        if request.method == 'POST':
            try:
                # Ambil input dari form
                data = {
                    'Luas Bangunan': float(request.form['LB']),
                    'Luas Tanah': float(request.form['LT']),
                    'Kamar Tidur': int(request.form['bedrooms']),
                    'Kamar Mandi': int(request.form['bathrooms']),
                    'Jumlah Lantai': int(request.form['floors']),
                    'Lokasi': request.form['Lokasi'],
                    'Kondisi': request.form['property_condition'],
                    'Furnitur': request.form['furnishing'],
                    'Garasi': int(request.form['garages']),
                    'Tahun': 2023
                }

                if data['Lokasi'] not in lokasi_valid:
                    raise ValueError(f"Lokasi '{data['Lokasi']}' tidak valid.")

                input_df = pd.DataFrame([data])[input_columns]
                input_df_original = input_df.rename(columns=reverse_mapping)

                input_processed = preprocessor.transform(input_df_original)
                prediksi = model.predict(input_processed)[0]

                prediction = '{:,.2f}'.format(prediksi[0])
                input_data = data

            except Exception as e:
                error = f"Terjadi kesalahan saat prediksi: {str(e)}"

        return render_template('index.html',
                               lokasi_options=lokasi_valid,
                               tabel=tabel_data.to_dict(orient='records'),
                               prediction=prediction,
                               input_data=input_data,
                               error=error)

    except Exception as e:
        return render_template('index.html', error=f"Application error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi harga rumah berdasarkan input form.
    Returns:
        dict: Hasil prediksi dalam format JSON
    """
    try:
        # Mengambil dan memvalidasi input dari form
        input_data = {
            'Luas Bangunan': float(request.form['LB']),
            'Luas Tanah': float(request.form['LT']),
            'Kamar Tidur': int(request.form['bedrooms']),
            'Kamar Mandi': int(request.form['bathrooms']),
            'Jumlah Lantai': int(request.form['floors']),
            'Lokasi': request.form['Lokasi'],
            'Kondisi': request.form['property_condition'],
            'Furnitur': request.form['furnishing'],
            'Garasi': int(request.form['garages']),
            'Tahun': 2023  # Default tahun saat ini
        }

        # Definisi kolom input untuk prediksi
        input_columns = [
            'Luas Bangunan', 'Luas Tanah', 'Kamar Tidur', 'Kamar Mandi',
            'Jumlah Lantai', 'Lokasi', 'Kondisi', 'Furnitur', 'Garasi', 'Tahun'
        ]
        
        # Mapping nama kolom Indonesia ke English untuk preprocessing
        column_mapping = {
            'Luas Bangunan': 'building_size_m2',
            'Luas Tanah': 'land_size_m2',
            'Kamar Tidur': 'bedrooms',
            'Kamar Mandi': 'bathrooms',
            'Jumlah Lantai': 'floors',
            'Lokasi': 'district',
            'Kondisi': 'property_condition',
            'Furnitur': 'furnishing',
            'Garasi': 'garages',
            'Tahun': 'year'
        }

        # Membuat DataFrame dan preprocessing
        input_df = pd.DataFrame([input_data])[input_columns]
        input_df_original = input_df.rename(columns=column_mapping)
        
        # Melakukan prediksi
        input_processed = preprocessor.transform(input_df_original)
        hasil_prediksi = model.predict(input_processed)[0]

        # Format hasil prediksi dengan pemisah ribuan dan 2 desimal
        return {
            'prediction': '{:,.2f}'.format(hasil_prediksi[0]),
            'status': 'success'
        }

    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }, 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
