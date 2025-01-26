import os, argparse, json
from hepinfo.util import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def train_model(args):

    from hepinfo.models.BinaryMI import BinaryMI
    from hepinfo.models.DebiasClassifier import DebiasClassifier

    # load data
    x, y, s, test, correlation, agreement_test = load_tau_data(args.data_path)

    # prepare agreement data
    agreement_weight = agreement_test["weight"]
    agreement_signal = agreement_test["signal"]
    agreement_test_feature = agreement_test.drop(columns=["id", "signal", "SPDhits", "weight"]).to_numpy()
    mass = correlation["mass"]
    correlation = correlation.drop(['id', 'mass', 'SPDhits'], axis=1).to_numpy()

    # get splits
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(x, y)

    # get results
    results = {
        "auc": [],
        "ks_value": [],
        "cvm": []
    }

    for i, (train_index, test_index) in enumerate(skf.split(x, y)):

        # get splits
        X_train = x[train_index]
        X_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        S_train = s[train_index]
        S_test = s[test_index]

        # transform data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        agreement_test_feature_scale = scaler.transform(agreement_test_feature)
        correlation_scale = scaler.transform(correlation)

        with open(args.hp_file) as f:
            hps = json.load(f)
            if args.model_name == "BinaryMI":
                hps["input_shape"] = (X_train.shape[1],)
            hps["name"] = args.model_name

        # store the used hps
        with open(f"{args.model_path}/{args.run_name}/hps.json", "w") as f:
            json.dump(hps, f)

        # get the model and train the model
        if args.model_name == "BinaryMI":
            model = BinaryMI(**hps)
        elif args.model_name == "DebiasClassifier":
            model = DebiasClassifier(**hps)
        else:
            raise ValueError(f"Model name {args.model_name} not found!")

        history = model.fit(X_train, y_train, S_train)
        model.model.save(f"{args.model_path}/{args.run_name}/model-split-{i}.keras")

        with open(f"{args.model_path}/{args.run_name}/history-split-{i}.json", "w") as f:
            json.dump(history.history, f)

        # predict model performance
        if args.model_name == "BinaryMI": pred = model.predict_proba(X_test)[:,1]
        if args.model_name == "DebiasClassifier": pred = model.predict_proba(X_test)

        auc = roc_auc_score(y_test, pred)

        if args.model_name == "BinaryMI": pred_agree = model.predict_proba(agreement_test_feature_scale)[:,1]
        if args.model_name == "DebiasClassifier": pred_agree = model.predict_proba(agreement_test_feature_scale)

        ks_value = compute_ks(
            pred_agree[agreement_signal == 0],
            pred_agree[agreement_signal == 1],
            agreement_weight[agreement_signal == 0],
            agreement_weight[agreement_signal == 1]
        )

        if args.model_name == "BinaryMI": pred_correlation = model.predict_proba(correlation_scale)[:,1]
        if args.model_name == "DebiasClassifier": pred_correlation = model.predict_proba(correlation_scale).reshape(-1)

        cvm = compute_cvm(pred_correlation, mass)

        results["auc"].append(auc)
        results["ks_value"].append(ks_value)
        results["cvm"].append(cvm)

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
