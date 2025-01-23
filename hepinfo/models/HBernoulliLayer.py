import hls4ml
import hls4ml.utils
import hls4ml.converters


# setup custom layer
rev_config_template = """struct config{index} : nnet::bernoulli_config {{
    static const unsigned n_in = {n_in};
    const ap_ufixed<1,0> thr = 0.5;
}};\n"""

rev_function_template = 'nnet::bernoulli<{input_t}, {config}>({input}, {output});'
rev_include_list = ['nnet_utils/bernoulli.h']

class HBernoulli(hls4ml.model.layers.Layer):
    '''hls4ml implementation of the bernoulli layer'''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)


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
