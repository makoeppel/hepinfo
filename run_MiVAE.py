import matplotlib.pyplot as plt
import mplhep
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from hepinfo.models.MiVAE import MiVAE

mplhep.style.use('CMS')

# read in the data
normal_data = np.load('./data/normal_data.npy', allow_pickle=True)
abnormal_data = np.load('./data/abnormal_data.npy', allow_pickle=True)

# perform some pre-processing and split into train test
nPV = normal_data[:, 0]
nPV_abnormal = abnormal_data[:, 0]
X = normal_data[:, 1:]
abnormal_data = abnormal_data[:, 1:]

X_train, X_test, nPV_train, nPV_test = train_test_split(X, nPV, shuffle=True)
abnormal_data_train, abnormal_data_test, nPV_abnormal_train, nPV_abnormal_test = train_test_split(
    abnormal_data, nPV_abnormal, shuffle=True
)

scaler = MinMaxScaler()  # add robast scaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
abnormal_data_train_scaled = scaler.transform(abnormal_data_train)
abnormal_data_test_scaled = scaler.transform(abnormal_data_test)

use_quantflow = False
use_s_quark = True
use_qkeras = False
MiVAE_model = MiVAE(
    verbose=2,
    activation='tanh',
    use_s_quark=use_s_quark,
    use_qkeras=use_qkeras,
    use_quantflow=use_quantflow,
    use_batchnorm=False,
    use_quantized_sigmoid=False,
    batch_size=1024,
    beta_param=1,
    alpha=1,  # can be also set to random to have random alphas for each layer
    beta0=10,
    init_quantized_bits=16,  # can be set to random to have random bit size for each value
    input_quantized_bits='quantized_bits(16, 6, 0)',
    quantized_bits='quantized_bits(16, 6, 0, use_stochastic_rounding=True)',
    quantized_activation='quantized_relu(10, 6, use_stochastic_rounding=True, negative_slope=0.0)',
    drop_out=0.0,
    epoch=10,
    gamma=1,
    num_samples=10,
    hidden_layers=[32, 16],
    latent_dims=8,
    learning_rate=0.001,
    optimizer='Adam',
    patience=100,
    mi_loss=True,
    run_eagerly=False,
)
print('Fit gamma ', 0.1)
history = MiVAE_model.fit(X_train, nPV_train)

for layer in MiVAE_model.encoder.layers:
    if hasattr(layer, '_beta'):
        tf.print(layer._beta)

# get data
mean_abnormal_score = MiVAE_model.get_mean(abnormal_data_test_scaled).numpy()
mean_abnormal_score = np.sum(mean_abnormal_score**2, axis=1)
mean_normal_score = MiVAE_model.get_mean(X_test).numpy()
mean_normal_score = np.sum(mean_normal_score**2, axis=1)

# generate plots for the ROC curves
y_test = np.concatenate((np.ones(len(mean_abnormal_score)), np.zeros(len(mean_normal_score))))
fpr, tpr, thresholds = roc_curve(y_test, np.concatenate((mean_abnormal_score, mean_normal_score)))
auc_value = auc(fpr, tpr)
print(auc_value)

if use_quantflow:
    plt.plot(history.history['bops'], '.')
    plt.ylabel('BOPs')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.show()

if use_s_quark:
    plt.plot(history.history['ebops'], '.')
    plt.ylabel('EBOPs')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.show()
