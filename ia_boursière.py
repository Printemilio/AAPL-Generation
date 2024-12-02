import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam

data = yf.download('AAPL', start='2022-01-01', end='2024-03-04')
data.to_csv('donnees_boursieres_aapl.csv')
data_loaded = pd.read_csv('donnees_boursieres_aapl.csv')

data = data[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

taille_fenetre = 60
X, y = [], []
for i in range(taille_fenetre, len(data_scaled)):
    X.append(data_scaled[i-taille_fenetre:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential([
    LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=L1L2(l1=0.0001, l2=0.0001)),
    Dropout(0.2),
    LSTM(300, return_sequences=False, kernel_regularizer=L1L2(l1=0.0001, l2=0.0001)),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(14,5))
plt.plot(y_test, color='red', label='Prix réel des actions AAPL')
plt.plot(predicted_stock_price, color='blue', label='Prédictions des actions AAPL')
plt.title('Prédiction du prix des actions AAPL')
plt.xlabel('Temps')
plt.ylabel('Prix des actions AAPL')
plt.legend()
plt.show()

latest_data = yf.download('AAPL', period='15d', interval='1d')['Close'].values
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1))
latest_data_scaled = np.reshape(latest_data_scaled, (1, latest_data_scaled.shape[0], 1))

predicted_price_scaled = model.predict(latest_data_scaled)
predicted_price = scaler.inverse_transform(predicted_price_scaled)
print(f"Le prix prédit pour le prochain jour de trading est: {predicted_price[0][0]}")

data_recent = yf.download('AAPL', period='90d', interval='1d')
real_prices = data_recent['Close'].values
predicted_prices = []
for i in range(taille_fenetre, len(real_prices)):
    input_data = real_prices[i-taille_fenetre:i]
    input_data_scaled = scaler.transform(input_data.reshape(-1, 1))
    input_data_scaled = np.reshape(input_data_scaled, (1, taille_fenetre, 1))
    predicted_price_scaled = model.predict(input_data_scaled)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_prices.append(predicted_price[0, 0])

dates = data_recent.index[taille_fenetre:]

plt.figure(figsize=(16,8))
plt.plot(data_recent.index, real_prices, color='black', label='Prix réels')
plt.plot(dates, predicted_prices, color='orange', label='Prédictions')
plt.scatter(dates[-1], predicted_prices[-1], color='red', label='Dernière prédiction')
plt.title('Prédictions du prix des actions avec les prix réels')
plt.xlabel('Date')
plt.ylabel('Prix des actions')
plt.legend()
plt.xticks(rotation=45)
plt.show()
