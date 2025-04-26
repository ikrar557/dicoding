#!/usr/bin/env python
# coding: utf-8

# # Predictive Analytics - Microsoft Daily Stock Data (Forecasting)

# Dalam proyek ini, model predictive akan dibangun untuk membantu memprediksi harga saham Microsoft dengan target fitur Close karena dianggap paling mempresentasikan harga saham pada hari tersebut.

# ## 1. Import Libraries

# Pada tahapan ini akan di import semua library yang digunakan dalam project predictive analysis ini.

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import plotly.graph_objects as go

import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")


# ## 2. Data Loading

# Pada tahapan ini, dataset microsft daily stock dimuat untuk selanjutnya akan dilakukan analisa.

# In[2]:


df = pd.read_csv('MSFT_daily_stock_data.csv')


# In[3]:


df


# ## 3. Exploratory Data Analysis

# Pada tahapan ini kita akan menganalisa dataset yang kita gunakan dan melakukan eksplorasi lebih lanjut sebelum dilakukan proses pembangunan model awal.

# ### 3.1 Deskripsi Variabel

# In[4]:


df.info()


# In[5]:


df.describe()


# Hasil analisa awal menunjukan dataset 7 fitur yang berisi 1 feature object dan 6 feature numerikal

# ### 3.2 Menangani Missing Values

# In[6]:


df.isnull().sum()


# Hasil pengecekan missing values terlihat semua baris atau sampel data tidak memiliki nilai yang hilang, sehingga tidak perlu dilakukan pembersihan

# ### 3.3 Menangani Outliers

# Pada tahapan ini kita akan memeriksa outliers dari dataset yang digunakan

# In[7]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.title('Boxplot Harga Saham dan Volume')
plt.xticks(rotation=45)
plt.show()


# In[8]:


z_scores = np.abs(stats.zscore(df[['Open', 'High', 'Low', 'Close', 'Volume']]))
threshold = 3
outlier_z = (z_scores > threshold)

outlier_count_z = outlier_z.sum(axis=0)
print("Jumlah outlier menggunakan Z-Score:\n", outlier_count_z)


# In[9]:


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()

outlier_count_iqr = {col: detect_outliers_iqr(df, col) for col in ['Open', 'High', 'Low', 'Close', 'Volume']}
print("Jumlah outlier menggunakan IQR:\n", outlier_count_iqr)


# Berdasarkan pemeriksaan, tampak outlier yang muncul adalah outlier yang bisa dianggap wajar dikarenakan adanya fluktuasi harga saham pada saat-saat tertentu.

# ### 3.4 Univariate Analysis

# Pada tahapan ini tipe data untuk fitur `Date` akan diubah dulu menjadi `datetime` agar lebih sesuai

# In[10]:


df['Date'] = pd.to_datetime(df['Date'])


# Pada tahapan ini distribusi dari setiap fitur numerik akan dilihat.

# In[11]:


numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()


# Pada tahapan ini tren dari harga sama penutupan `Close` akan dilihat, terlihat pada hasil visualisasi tren harga terkadang stagnan pada tahun 2000-an meningkat secara signifikan

# In[12]:


plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], color='blue')
plt.title('Tren Harian Harga Penutupan (Close)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### 3.5 Multivariate Analysis

# Pada tahapa ini kita akan memeriksa korelasi pada setiap fitur numerikal, terlihat fitur seperti `Open`, `High`, `Low`, `Close`, dan `Adj Close` yang kemungkinan terjadi overlapped.

# In[13]:


correlation_matrix = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriks Korelasi Antar Fitur')
plt.show()


# Visualisasi diabwah untuk menampilkan korelasi antara `volume` dengan `close` menggunakan scatter plot.

# In[14]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='Volume', y='Close', data=df)
plt.title('Scatter Plot Volume vs Harga Penutupan')
plt.xlabel('Volume')
plt.ylabel('Close')
plt.show()


# Visualisasi berikut digunakan untuk melihat regresi linear antara `volume` dan dengan `close`, terlihat pada visualisasi korelasi sangat lemah

# In[15]:


sns.lmplot(x='Volume', y='Close', data=df, aspect=2, height=5, line_kws={'color': 'red'})
plt.title('Regresi Linear Volume vs Harga Penutupan')
plt.xlabel('Volume')
plt.ylabel('Harga Penutupan')
plt.show()


# ## 4. Data Preparation

# ### 4.1 Penetapan Index Waktu

# Data time series harus memiliki urutan waktu yang eksplisit, sehingga kolom `Date` dijadikan index untuk menjaga urutan data.

# In[16]:


df_model = df[['Date', 'Close']]
df_model = df_model.set_index('Date')


# ### 4.2 Normalisasi Data

# Pada tahapan ini akan dilakukan normalisasi agar dapat mempercepat proses pelatihan dan membantu model memahami pola dari data

# In[17]:


scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_model)


# In[18]:


df_scaled


# ### 4.3 Penentuan Sequence Data

# Menentukan model model belajar memprediksi harga pada hari ke-61 berdasarkan 60 hari sebelumnya.

# In[19]:


def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(df_scaled, sequence_length)



# ### 4.4 Data Splitting

# Data dibagi dengan rasio 80:20 untuk training dan testing agar model diuji pada data yang belum pernah dilihat sebelumnya.

# In[20]:


train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]


# ## 5. Modelling

# ### Callbacks

# Pada tahapan ini, fungsi callbacks `reduce_lr_callback` dibuat agar saat *training* learning rate pada model dapat disesuaikan secara otomatis saat model stagnan. Fungsi callback `early_stopping_callback` juga dibuat agar pelatihan dapat berhenti ketika evaluasi model tidak membaik, dengan begitu resource dapat lebih dihemat.

# In[21]:


early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)


# Pada tahapan awal, model `LSTM` dipilih untuk membangun analisis prediktif karena memang sesuai untuk analisa forecasting time series

# ### 5.1 LSTM Approach

# #### 5.1.1 Development

# Pada tahapan ini, model deep learning 3 lapis LSTM berurutan (64, 64, dan 32 unit) dibangun untuk menangkap pola jangka pendek dan panjang dalam data time series.  Model dikompilasi dengan optimizer Adam dan loss MSE untuk prediksi nilai.

# In[22]:


model_lstm = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=64, return_sequences=True),
    LSTM(units=32),
    Dense(units=25),
    Dense(units=1)
]) 

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.summary()


# Pada tahapan ini, model di latih menggunakan parameter-parameter yang sudah ditentukan, beserta fungsi callbacks yang sudah di buat sebelumnya

# In[23]:


history_lstm = model_lstm.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping_callback, reduce_lr_callback],
                    verbose=1)


# #### 5.1.2 Evaluation

# Pada tahapan ini, loss dari model akan di evaluasi terlebih dahulu.

# In[24]:


test_loss = model_lstm.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")


# Evalusasi menggunakan berbagai metrik dilakukan untuk melihat performa dari model yang digunakan, pada hasil akhir terlihat evaluasi pada metrik R-Squared menunjukan hasil yang hampir sempurna.

# In[25]:


lstm_prediction = model_lstm.predict(X_test)
lstm_prediction = scaler.inverse_transform(lstm_prediction)
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

tanggal_prediksi = df_model.index[-len(y_test):]

comparison = pd.DataFrame({
    'Tanggal': tanggal_prediksi,
    'Harga Saham Asli': y_test_true.flatten(),
    'Harga Saham Prediksi': lstm_prediction.flatten()
})

mae = mean_absolute_error(y_test_true, lstm_prediction)
mse = mean_squared_error(y_test_true, lstm_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_true, lstm_prediction)

print(comparison.head())
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared Score: {r2:.4f}')


# Tahapan evalusasi visual digunakan agar dapat membandingkan nilai saham asli dengan nilai saham prediksi secara jelas dan keseluruhan.

# In[26]:


data = pd.DataFrame({
    'Tanggal': comparison['Tanggal'],
    'True Closing Price': comparison['Harga Saham Asli'],
    'Predicted Closing Price': comparison['Harga Saham Prediksi']
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['Tanggal'],
    y=data['True Closing Price'],
    mode='lines',
    name='Harga Saham Asli',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=data['Tanggal'],
    y=data['Predicted Closing Price'],
    mode='lines',
    name='Harga Saham Prediksi',
    line=dict(color='red')
))

fig.update_layout(
    title="Prediksi Harga Saham vs Harga Asli",
    xaxis_title="Tanggal",
    yaxis_title="Harga Saham",
    template="plotly_dark",
    xaxis=dict(tickformat="%Y-%m-%d"),
    showlegend=True
)

fig.show()


# ### 5.2 GRU Approach

# #### 5.2.1 Development

# Pada tahapan ini, model deep learning satu lapis GRU dengan 50 unit untuk menangkap pola dalam data time,langsung diikuti oleh satu Dense layer dengan 1 unit untuk regresi. Model dikompilasi dengan optimizer Adam dan loss MSE untuk memprediksi nilai.

# In[27]:


model_gru = Sequential()
model_gru.add(GRU(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model_gru.add(Dense(units=1))

model_gru.compile(optimizer='adam', loss='mean_squared_error')

model_gru.summary()


# Pada tahapan ini, model di latih menggunakan parameter-parameter yang sudah ditentukan, beserta fungsi callbacks yang sudah di buat sebelumnya

# In[28]:


history_gru = model_gru.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping_callback, reduce_lr_callback],
                    verbose=1)


# #### 5.2.2 Evaluation

# Pada tahapan ini, loss dari model akan di evaluasi terlebih dahulu.

# In[29]:


test_loss = model_gru.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")


# Evalusasi menggunakan berbagai metrik dilakukan untuk melihat performa dari model yang digunakan, pada hasil akhir terlihat evaluasi pada metrik R-Squared menunjukan hasil yang lebih baik daripada model LSTM waluapun hanya berbeda tipis.

# In[30]:


gru_prediction = model_gru.predict(X_test)

gru_prediction = scaler.inverse_transform(gru_prediction)

y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"Shape of predicted_stock_price: {gru_prediction.shape}")
print(f"Shape of y_test_true: {y_test_true.shape}")

tanggal_prediksi = df_model.index[-len(y_test):]

comparison = pd.DataFrame({
    'Tanggal': tanggal_prediksi,
    'Harga Saham Asli': y_test_true.flatten(),
    'Harga Saham Prediksi': gru_prediction.flatten()
})

mae = mean_absolute_error(y_test_true, gru_prediction)
mse = mean_squared_error(y_test_true, gru_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_true, gru_prediction)

print(comparison.head())
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared Score: {r2:.4f}')


# Tahapan evalusasi visual digunakan agar dapat membandingkan nilai saham asli dengan nilai saham prediksi secara jelas dan keseluruhan.

# In[31]:


data = pd.DataFrame({
    'Tanggal': comparison['Tanggal'],
    'True Closing Price': comparison['Harga Saham Asli'],
    'Predicted Closing Price': comparison['Harga Saham Prediksi']
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['Tanggal'],
    y=data['True Closing Price'],
    mode='lines',
    name='Harga Saham Asli',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=data['Tanggal'],
    y=data['Predicted Closing Price'],
    mode='lines',
    name='Harga Saham Prediksi',
    line=dict(color='red')
))

fig.update_layout(
    title="Prediksi Harga Saham vs Harga Asli",
    xaxis_title="Tanggal",
    yaxis_title="Harga Saham",
    template="plotly_dark",
    xaxis=dict(tickformat="%Y-%m-%d"),
    showlegend=True
)

fig.show()


# ### 5.3 Comparing models

# Pada tahapan ini, kedua model yaitu LSTM dan GRU akan dibandingkan secara visualisasi dengan harga saham aslinya

# In[32]:


data_compare = pd.DataFrame({
    'Tanggal': tanggal_prediksi,
    'Harga Asli': y_test_true.flatten(),
    'Prediksi LSTM': lstm_prediction.flatten(),
    'Prediksi GRU': gru_prediction.flatten()
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data_compare['Tanggal'],
    y=data_compare['Harga Asli'],
    mode='lines',
    name='Harga Saham Asli',
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=data_compare['Tanggal'],
    y=data_compare['Prediksi LSTM'],
    mode='lines',
    name='Prediksi LSTM',
    line=dict(color='red', width=2)
))

fig.add_trace(go.Scatter(
    x=data_compare['Tanggal'],
    y=data_compare['Prediksi GRU'],
    mode='lines',
    name='Prediksi GRU',
    line=dict(color='green', width=2)
))

fig.update_layout(
    title="Perbandingan Harga Saham Asli vs Prediksi (LSTM vs GRU)",
    xaxis_title="Tanggal",
    yaxis_title="Harga Saham",
    template="plotly_dark",
    xaxis=dict(tickformat="%Y-%m-%d"),
    showlegend=True
)

fig.show()


# Terlihat pada visualisasi diatas, model GRU lebih akurat dalam memprediksi harga asli saham daripada model LSTM. Hal ini bisa disebabkan window sequence time yang diambil hanya jangka pendek saja, yaitu 60 hari. Sedangkan model LSTM lebih optimal jika digunakan pada jangka panjang. Selain itu bisa juga model LSTM memerlukan tuning agar outputnya lebih maksimal
