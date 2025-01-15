import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from hepinfo.models.BinaryMI import BinaryMI
from sklearn.metrics import roc_auc_score


# read in the data
normal_data = np.load("../data/normal_data.npy", allow_pickle=True)
abnormal_data = np.load("../data/abnormal_data.npy", allow_pickle=True)
nPV_normal = normal_data[:,0]
nPV_abnormal = abnormal_data[:,0]
normal_data = normal_data[:,1:]
abnormal_data = abnormal_data[:,1:]

y = np.concatenate((np.ones(len(abnormal_data)), np.zeros(len(normal_data))))
X = np.concatenate((abnormal_data, normal_data))
S = np.concatenate((nPV_abnormal, nPV_normal))

X_train, X_test, y_train, y_test, S_train, S_test  = train_test_split(X, y, S, random_state=42)

# train the transfer model
use_quantflow = False
use_s_quark = False
use_qkeras = True
BMI = BinaryMI(
    kernel_regularizer = 0.01,
    batch_size=32,
    hidden_layers = [64, 32, 16],
    quantized_position=[False, True, False],
    validation_size=0.1,
    use_quantflow=use_quantflow,
    use_s_quark=use_s_quark,
    use_qkeras=use_qkeras,
    init_quantized_bits=16,
    input_quantized_bits="quantized_bits(16, 6, 0)",
    quantized_bits="quantized_bits(16, 6, 0, use_stochastic_rounding=True)",
    quantized_activation="quantized_relu(10, 6, use_stochastic_rounding=True, negative_slope=0.0)",
    alpha=1e-6,
    beta0=1e-5,
    epoch=1,
    input_shape=(X_train.shape[1],),
    verbose=2,
    learning_rate_decay_rate=0.001,
    gamma=0.3,
    print_summary=True,
    last_layer_size=2,
    run_eagerly=False
)
history = BMI.fit(x_train=X_train, y_train=y_train, s_train=S_train)
if use_quantflow: BMI.model.save('binaryMI_quantflow.keras')
if use_s_quark: BMI.model.save('binaryMI_squark.keras')
if use_qkeras: BMI.model.save('binaryMI_qkeras.keras')
if not use_qkeras and not use_quantflow and not use_s_quark: BMI.model.save('binaryMI_full.keras')

y_pred = BMI.predict_proba(X_test)[:,1]
auc_value = roc_auc_score(y_test, y_pred)
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
