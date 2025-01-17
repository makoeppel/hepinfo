from tensorflow.keras.models import load_model
from hepinfo.util import MILoss
from hepinfo.models.BinaryMI import BinaryMI
import numpy as np

import hls4ml
import hls4ml.utils
import hls4ml.converters

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from hepinfo.models.BinaryMI import BinaryMI
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
    static const unsigned thr = 0.5;
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

model = load_model("trained_models/binaryMI_qkeras.keras", custom_objects=custom_objects)

hmodel = hls4ml.converters.convert_from_keras_model(
    model,
    backend='Vitis',
    output_dir='binaryMI-HLS4ML',
    io_type='io_parallel',
    hls_config={'Model': {'Precision': 'ap_int<6>', 'ReuseFactor': 1}}
)

# compile and test the model
# normal_data = np.load('./data/normal_data.npy', allow_pickle=True)
# abnormal_data = np.load('./data/abnormal_data.npy', allow_pickle=True)
# nPV_normal = normal_data[:, 0]
# nPV_abnormal = abnormal_data[:, 0]
# normal_data = normal_data[:, 1:]
# abnormal_data = abnormal_data[:, 1:]

# y = np.concatenate((np.ones(len(abnormal_data)), np.zeros(len(normal_data))))
# X = np.concatenate((abnormal_data, normal_data))
# S = np.concatenate((nPV_abnormal, nPV_normal))

# X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, random_state=42)
# scaler = MinMaxScaler()  # add robast scaler
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# hmodel.compile()

# print(len(X_test))
# X_test = np.ascontiguousarray(X_test)
# print(len(X_test))
# print(X_test)
# y_pred = hmodel.predict(X_test)
# auc_value = roc_auc_score(y_test, y_pred)
# print(auc_value)

# build the model
hmodel.build()

hls4ml.report.read_vivado_report('binaryMI-HLS4ML')
