import hls4ml
import hls4ml.utils
import hls4ml.converters


# Fetch a keras model from our example repository
# This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

# You can print the configuration to see some default parameters
print(config)

# Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example models if you want to explore more
hls4ml.utils.fetch_example_list()
