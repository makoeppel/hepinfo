import os, argparse

parser = argparse.ArgumentParser("Run load trained models")
parser.add_argument("project_path", help="Output path for the HLS4ML project.", type=str)
parser.add_argument("model_path", help="Model which should be load.", type=str)
args = parser.parse_args()

if os.path.isdir(args.project_path):
    raise ValueError(f'The project path {args.project_path} is not empty.')


from tensorflow import keras
from tensorflow.keras.models import load_model

from hepinfo.util import MILoss
from hepinfo.models.BinaryMI import BinaryMI
from hepinfo.models.qkerasV3 import QDense, QActivation, quantized_bits

import numpy as np

import hls4ml
import hls4ml.utils
import hls4ml.converters

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler


# hls4ml layer implementation
class HBernoulli(hls4ml.model.layers.Layer):
    '''hls4ml implementation of the bernoulli layer'''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

# Parser for converter
def parse_bernoulli_layer(keras_layer, input_names, input_shapes, data_reader, thr=0.5):
    layer = {}
    layer['class_name'] = 'BernoulliSampling'
    layer['name'] = 'bernoulli'#keras_layer['config']['name']
    layer['n_in'] = input_shapes[0][1]
    layer['thr'] = thr

    if input_names is not None:
        layer['inputs'] = input_names

    return layer, [shape for shape in input_shapes[0]]

rev_config_template = """struct config{index} : nnet::bernoulli_config {{
    static const unsigned n_in = {n_in};
    const ap_ufixed<1,0> thr = 0.5;
}};\n"""

rev_function_template = 'nnet::bernoulli<{input_t}, {config}>({input}, {output});'
rev_include_list = ['nnet_utils/bernoulli.h']


class HBernoulliConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HBernoulli)
        self.template = rev_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class HBernoulliFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HBernoulli, include_header=rev_include_list)
        self.template = rev_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)

# Register the converter for custom Keras layer
hls4ml.converters.register_keras_layer_handler('BernoulliSampling', parse_bernoulli_layer)

# Register the hls4ml's IR layer
hls4ml.model.layers.register_layer('BernoulliSampling', HBernoulli)

# Register the optimization passes (if any)
backend = hls4ml.backends.get_backend('Vitis')

# Register template passes for the given backend
backend.register_template(HBernoulliConfigTemplate)
backend.register_template(HBernoulliFunctionTemplate)

# Register HLS implementation
backend.register_source('/data/hlssynt-users/makoppel/hepinfo/tb/bernoulli.h')

custom_objects = {
    "MILoss": MILoss
}

# load data
normal_data = np.load('./data/normal_data.npy', allow_pickle=True)
abnormal_data = np.load('./data/abnormal_data.npy', allow_pickle=True)
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

# train the model
BMI = BinaryMI(
    kernel_regularizer=0.0,
    batch_size=1024,
    hidden_layers=[64, 32, 16],
    quantized_position=[False, True, False],
    validation_size=0.1,
    num_samples=10,
    batch_normalisation=False,
    use_quantized_sigmoid=True,
    activation_nonbinary="relu",
    bits_bernoulli_sigmoid=8,
    use_quantflow=False,
    use_s_quark=False,
    use_qkeras=False,
    init_quantized_bits=16,
    input_quantized_bits='quantized_bits(16, 6, 1)',
    quantized_bits='quantized_bits(16, 6, 1, use_stochastic_rounding=True)',
    quantized_activation='quantized_relu(16, 6, use_stochastic_rounding=True, negative_slope=0.0)',
    # acitvation_last_layer='quantized_sigmoid(16, 6, use_stochastic_rounding=True)',
    acitvation_last_layer='sigmoid',
    alpha=1e-6,
    beta0=1e-5,
    epoch=10,
    input_shape=(X_train.shape[1],),
    verbose=2,
    learning_rate=0.01,
    gamma=1,
    print_summary=True,
    last_layer_size=1,
    run_eagerly=False,
)

# # fit the model
# history = BMI.fit(x_train=X_train, y_train=y_train, s_train=S_train)
# BMI.model.save(args.model_path)

# # test the model
# y_pred = BMI.predict_proba(X_test)
# auc_value = roc_auc_score(y_test, y_pred)
# print("Kears model: ", auc_value)

# load the model
model = load_model(args.model_path, custom_objects=custom_objects)

# test the model
y_pred = model.predict(X_test)[0]
auc_value = roc_auc_score(y_test, y_pred)
print("Kears model: ", auc_value)

hmodel = hls4ml.converters.convert_from_keras_model(
    model,
    backend='Vitis',
    output_dir=args.project_path
)

# # test the HLS4ML model
# hmodel.compile()
# X_test = np.ascontiguousarray(X_test)
# y_pred = hmodel.predict(X_test)[0]
# auc_value = roc_auc_score(y_test, y_pred)
# print("HLS4ML model: ", auc_value)

# build the model
hmodel.build(vsynth=True)

hls4ml.report.read_vivado_report(args.project_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run load trained models")
    parser.add_argument("model_path", help="Model which should be load.", type=str)
    parser.add_argument("bernoulli_path", help="Absolute path to the bernoulli.h layer.", type=str)
    args = parser.parse_args()

    if os.path.isdir(args.project_path):
        raise ValueError(f'The project path {args.project_path} is not empty.')

    test_model(args)