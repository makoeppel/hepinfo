from tensorflow import keras
from tensorflow.keras.models import load_model

from hepinfo.models.qkerasV3 import QDense, QActivation, quantized_bits

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

# Define the QKeras model
# Define the input layer explicitly
input_layer = keras.layers.Input(shape=(X_train.shape[1],))  

# Define the QKeras model using Functional API (fixing Sequential's issue)
import qkeras
x = qkeras.qlayers.QDense(64)(input_layer)
x = qkeras.qlayers.QActivation('quantized_relu(16, 0)')(x)

x = qkeras.qlayers.QDense(32)(x)
x = qkeras.qlayers.QActivation('quantized_relu(16, 0)')(x)

x = qkeras.qlayers.QDense(16)(x)
x = qkeras.qlayers.QActivation('quantized_relu(16, 0)')(x)

output_layer = qkeras.qlayers.QDense(1, activation="sigmoid")(x)

# NOTE: you can also test the custom qkerasV3
# x = QDense(64)(input_layer)
# x = QActivation('quantized_relu(16, 0)')(x)

# x = QDense(32)(x)
# x = QActivation('quantized_relu(16, 0)')(x)

# x = QDense(16)(x)
# x = QActivation('quantized_relu(16, 0)')(x)

# x = QDense(1)(x)
# x = QActivation("quantized_bits(20, 5)")(x)

# output_layer = QActivation("sigmoid")(x)


# Create the model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=10)


# test the model
y_pred = model.predict(X_test)
auc_value = roc_auc_score(y_test, y_pred)
print("Kears model: ", auc_value)

# save the model
model.save('../trained_models/test_qkeras.keras')

# load the model
model = load_model("../trained_models/test_qkeras.keras")

# test the model
y_pred = model.predict(X_test)[0]
auc_value = roc_auc_score(y_test, y_pred)
print("Kears model: ", auc_value)

hmodel = hls4ml.converters.convert_from_keras_model(
    model,
    backend='Vitis',
    output_dir='testQkeras-HLS4ML-V3'
)

# test the HLS4ML model
hmodel.compile()
X_test = np.ascontiguousarray(X_test)
y_pred = hmodel.predict(X_test)
auc_value = roc_auc_score(y_test, y_pred)
print("HLS4ML model: ", auc_value)
print(np.unique(y_pred))
exit()

# build the model
hmodel.build()

hls4ml.report.read_vivado_report('testQkeras-HLS4ML-V3')
