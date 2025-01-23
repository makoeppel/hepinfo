from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np

import hls4ml
import hls4ml.utils
import hls4ml.converters

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

# Register the optimization passes (if any)
backend = hls4ml.backends.get_backend('Vitis')

# load data
normal_data = np.load('./../data/normal_data.npy', allow_pickle=True)
abnormal_data = np.load('./../data/abnormal_data.npy', allow_pickle=True)
nPV_normal = normal_data[:, 0]
nPV_abnormal = abnormal_data[:, 0]
normal_data = normal_data[:, 1:]
abnormal_data = abnormal_data[:, 1:]

y = np.concatenate((np.ones(len(abnormal_data)), np.zeros(len(normal_data))))
X = np.concatenate((abnormal_data, normal_data))
S = np.concatenate((nPV_abnormal, nPV_normal))

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, random_state=42)
scaler = MinMaxScaler()  # add robast scaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# keras model
input_layer = keras.layers.Input((X_train.shape[1],))
x = keras.layers.Dense(64, activation="relu")(input_layer)
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dense(16, activation="relu")(x)
output_layer = keras.layers.Dense(1, activation="sigmoid")(x)

# Create the model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=1)

# test the model
y_pred = model.predict(X_test)
auc_value = roc_auc_score(y_test, y_pred)
print("Kears model: ", auc_value)

# save the model
model.save('../trained_models/test_keras.h5')

# load the model
model = load_model("../trained_models/test_keras.h5")

# test the model
y_pred = model.predict(X_test)
auc_value = roc_auc_score(y_test, y_pred)
print("Kears model: ", auc_value)

hmodel = hls4ml.converters.convert_from_keras_model(
    model,
    backend='Vitis',
    output_dir='testkeras-HLS4ML-V3'
)

# test the HLS4ML model
hmodel.compile()
X_test = np.ascontiguousarray(X_test)
y_pred = hmodel.predict(X_test)
auc_value = roc_auc_score(y_test, y_pred)
print("HLS4ML model: ", auc_value)

# build the model
hmodel.build()

hls4ml.report.read_vivado_report('testkeras-HLS4ML-V3')
