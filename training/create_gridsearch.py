import os, argparse, json


def create_run_scripts(args):

    # load hps
    with open(args.hp_file) as f:
        hps = json.load(f)

    if args.model_name == "BinaryMI":
        counter = 0
        for hidden_layers, quantized_position in \
            zip([[64, 32, 16], [32, 16], [16, 8]], \
                [[False, True, False], [True, False], [True, False]]):
            for gamma in [0, 1, 10, 100]:
                hps["hidden_layers"] = hidden_layers
                hps["quantized_position"] = quantized_position
                hps["gamma"] = gamma

                with open(f"{args.run_name}-{args.model_name}/hps-{counter}.json", "w") as f:
                    json.dump(hps, f)
                counter += 1

    if args.model_name == "DebiasClassifier":
        counter = 0
        for hidden_layers in [[64, 32, 16], [32, 16], [16, 8]]:
            for bias_layers in [[32, 16], [10]]:
                for gamma in [0, 1, 10, 100]:
                    hps["hidden_layers"] = hidden_layers
                    hps["bias_layers"] = bias_layers
                    hps["gamma"] = gamma

                    with open(f"{args.run_name}-{args.model_name}/hps-{counter}.json", "w") as f:
                        json.dump(hps, f)
                    counter += 1

    # create run file
    run_file = ""
    for i in range(counter):
        run_file += f"python ../train_supervised_model.py ../{args.data_path} hps-{i}.json ../{args.model_path} {args.run_name}-{i} {args.model_name} &\n"
    with open(f"{args.run_name}-{args.model_name}/run.sh", "w") as f:
        f.write(run_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run load trained models")
    parser.add_argument("data_path", help="Input path for the data.", type=str)
    parser.add_argument("hp_file", help="Input file for the base HPs.", type=str)
    parser.add_argument("model_path", help="Output path where the model and results should be stored.", type=str)
    parser.add_argument("run_name", help="Name of the run.", type=str)
    parser.add_argument("model_name", help="Which model should be trained.", type=str)
    args = parser.parse_args()

    if not os.path.exists(f"{args.run_name}-{args.model_name}/"):
        os.makedirs(f"{args.run_name}-{args.model_name}")
    else:
        raise ValueError(f"Path {args.run_name} exists.")

    create_run_scripts(args)
