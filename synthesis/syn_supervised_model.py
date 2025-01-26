import os, argparse


def synthesis_model(args):

    from tensorflow.keras.models import load_model

    from hepinfo.util import MILoss
    from hepinfo.models.HBernoulliLayer import parse_bernoulli_layer, HBernoulli, HBernoulliConfigTemplate, HBernoulliFunctionTemplate

    import hls4ml
    import hls4ml.utils
    import hls4ml.converters


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
    backend.register_source(args.bernoulli_path)

    custom_objects = {
        "MILoss": MILoss
    }

    # load the model
    model = load_model(args.model_path, custom_objects=custom_objects)

    hmodel = hls4ml.converters.convert_from_keras_model(
        model,
        backend='Vitis',
        output_dir=args.project_path
    )

    # build the model
    hmodel.build(vsynth=True)

    hls4ml.report.read_vivado_report(args.project_path)

    # TODO: add results for this (resource usage, AUC value etc.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run load trained models")
    parser.add_argument("project_path", help="Output path for the HLS4ML project.", type=str)
    parser.add_argument("model_path", help="Model which should be load.", type=str)
    parser.add_argument("bernoulli_path", help="Absolute path to the bernoulli.h layer.", type=str)
    args = parser.parse_args()

    if os.path.isdir(args.project_path):
        raise ValueError(f'The project path {args.project_path} is not empty.')

    synthesis_model(args)
