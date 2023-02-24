# This code find numberic relationship between to sets of data of equal length which is in this example n+1.
import numpy as np
from keras import Sequential
from keras.layers import Dense
import math

input = 1000

def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return math.ceil(float(num)) 
    return math.floor(float(num))

model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
ys = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)

model.fit(xs, ys, epochs=2000)

result = model.predict([input])[-1][-1]
rounded_result = proper_round(str(result))

print("\nInput:", input, "\nPrediction:", rounded_result, "\n")
