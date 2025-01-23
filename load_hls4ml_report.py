import argparse

import hls4ml
import hls4ml.utils
import hls4ml.converters

parser = argparse.ArgumentParser("Read HLS4ML Reports")
parser.add_argument("path", help="Path to the compilation.", type=str)
args = parser.parse_args()

hls4ml.report.read_vivado_report(args.path)
