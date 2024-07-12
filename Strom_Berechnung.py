import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras import models, layers, backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

# Установка случайного состояния для воспроизводимости
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_random_seed()

# Функция для разбиения на тренировочный и тестовый наборы
def split_dataset(data):
    train, test = data[:7885], data[7885:]
    return train, test

# Функция для преобразования истории в входные и выходные данные
def split_sequence(sequence, n_input, n_output):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_input
        out_end_ix = end_ix + n_output
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Функция для создания прогноза
def forecast(model, history, n_input):
    input_x = np.array(history[-n_input:])
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    yhat = model.predict(input_x, verbose=0)
    return yhat[0]

# Функция для тренировки модели
def build_model(train, n_input, n_output):
    train_x, train_y = split_sequence(train, n_input, n_output)
    verbose, epochs, batch_size = 0, 60, 16 
    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
    model = models.Sequential()
    model.add(layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(layers.RepeatVector(n_output))
    model.add(layers.LSTM(200, activation='relu', return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# Функция для оценки модели
def evaluate_model(train, test, n_input, n_output):
    model = build_model(train, n_input, n_output)
    history = [x for x in train]
    predictions = []
    for i in range(len(test) - n_input):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = np.array(predictions)
    actual = np.array([test[i:i+n_output, 1] for i in range(len(test) - n_input)])
    score, scores = evaluate_forecasts(actual, predictions)
    return score, scores, predictions

# Функция для оценки прогнозов
def evaluate_forecasts(actual, predicted):
    scores = []
    for i in range(predicted.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    s = 0
    for row in range(predicted.shape[0]):
        for col in range(predicted.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (predicted.shape[0] * predicted.shape[1]))
    return score, scores

# Загрузка данных
dataset = pd.read_csv('C:/Studium/Computing Science/Hospital_Dataset.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
dataset.columns = ['temperature', 'Global_active_power']

# Разделение данных на тренировочный и тестовый наборы
train, test = split_dataset(dataset.values)

# Параметры модели
n_input = 24  
n_output = 24  

# Оценка модели
score, scores, predictions = evaluate_model(train, test, n_input, n_output)

# Формирование списка дат для вывода
dates = dataset.index[7885 + n_input:7885 + n_input + len(predictions)]

# Вывод предсказаний в файл
output_file = 'C:/Studium/Computing Science/predictions.csv'
with open(output_file, 'w') as f:
    f.write('Date,Predictions\n')
    for date, pred in zip(dates, predictions):
        f.write(f"{date},{pred[0]}\n") 

print(f"Predictions saved to {output_file}")

# Прогноз на следующий период (24 часа) после окончания тестовых данных
next_prediction = forecast(build_model(train, n_input, n_output), list(test[-n_input:]), n_input)
print(f"Prediction for the next 24 hours: {next_prediction}")

# Построение графика ошибок
hours = ['hour ' + str(i) for i in range(1, 25)]
plt.plot(hours, scores, marker='o', label='lstm')
plt.show()