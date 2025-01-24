import os, argparse, json
from hepinfo.util import *
from sklearn.metrics import roc_auc_score


def train_model(args):

    from hepinfo.models.BinaryMI import BinaryMI
    from hepinfo.models.DebiasClassifier import DebiasClassifier

    train_vali_data, agreement_data, meta_data = load_tau_data(args.data_path)
    X_train, X_test, y_train, y_test, S_train, S_test = train_vali_data
    agreement_weight, agreement_signal, agreement_test_feature = agreement_data
    test, correlation, scaler = meta_data

    with open(args.hp_file) as f:
        hps = json.load(f)
        hps["input_shape"] = (X_train.shape[1],)
        hps["name"] = args.model_name

    # stor the used hps
    with open(f"{args.model_path}/{args.run_name}/hps.json", "w") as f:
        json.dump(hps, f)

    # get the model and train the model
    if args.model_name == "BinaryMI":
        model = BinaryMI(**hps)
    elif args.model_name == "DebiasClassifier":
        model = DebiasClassifier(**hps)
    else:
        raise ValueError(f"Model name {args.model_name} not found!")

    history = model.fit(x_train=X_train, y_train=y_train, s_train=S_train)
    model.model.save(f"{args.model_path}/{args.run_name}/model.keras")

    with open(f"{args.model_path}/{args.run_name}/history.json", "w") as f:
        json.dump(history.history, f)

    # predict model performance
    pred = model.predict_proba(X_test)

    auc = roc_auc_score(y_test, pred[:,1])

    pred_agree = model.predict_proba(agreement_test_feature)

    ks_value = compute_ks(
        pred_agree[:,1][agreement_signal == 0],
        pred_agree[:,1][agreement_signal == 1],
        agreement_weight[agreement_signal == 0],
        agreement_weight[agreement_signal == 1]
    )

    check_correlation_features = correlation.drop(['id', 'mass', 'SPDhits'], axis=1)
    pred_correlation = model.predict_proba(check_correlation_features)

    cvm = compute_cvm(pred_correlation[:,1], correlation['mass'])

    results = {
        "auc": auc,
        "ks_value": ks_value,
        "cvm": cvm
    }

    with open(f"{args.model_path}/{args.run_name}/results.json", "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run load trained models")
    parser.add_argument("data_path", help="Input path for the data.", type=str)
    parser.add_argument("hp_file", help="Input file for the HPs.", type=str)
    parser.add_argument("model_path", help="Output path where the model and results should be stored.", type=str)
    parser.add_argument("run_name", help="Name of the run.", type=str)
    parser.add_argument("model_name", help="Which model should be trained.", type=str)
    args = parser.parse_args()

    if not os.path.exists(f"{args.model_path}/{args.run_name}"):
        os.makedirs(f"{args.model_path}/{args.run_name}")
    else:
        raise ValueError(f"Path {args.model_path}/{args.run_name} exists.")

    train_model(args)
