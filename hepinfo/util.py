import h5py, keras, os

import numpy as np
import pandas as pd
import awkward as ak
import tensorflow as tf

from joblib import dump, load

from keras.api import ops

from hepinfo.models.qkerasV3 import quantized_sigmoid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc


def readFromAnomalyh5(inputfile, process, object_ranges='default1', moreInfo=None, verbosity=0):
    """
    Reads data from h5s that either contain multiple signal processes or background that are loaded individually

    inputfile -- h5 file containing either background events or multiple signal processes
    process -- the name of the (signal) process, in case of reading a background file, please set this to 'background'
    object ranges -- dictionary containing the keys "met", "jets", "muons" and "egs" and for each key a tuple containing
                     the range of the corresponding objects in the data array of the h5 file. For "met" it should be given
                     only the index (one number, no tuple), because by definition there can be only one MET object
                     Alternatively it can be set to the string "default1" (4 egs, 4 muons, 10 jets) or
                     "default2" (12 egs, 8 muons, 12 jets)
    moreInfo -- more info about the data that is included in the output
    verbosity -- verbosity of this function
    """

    if verbosity > 0:
        print('Reading anomaly team preprocessed file at ' + inputfile + ' for process ' + process + '.')

    # constructing the information dict
    # some things will be automatically filled here
    # the input "moreInfo" can be used to pass more information
    # this information will have priority over automatically set entries
    infoDict = {}
    infoDict['input'] = inputfile

    # preparing lists to store L1 bit info
    L1bits_labels = []
    L1bits = []

    eventData = {}  # initialize as empty in case file does not contain "event_info" key

    # reading the intput file
    if verbosity > 0:
        print('Starting to read input file...')
    with h5py.File(inputfile, 'r') as h5f2:
        if process == 'background':
            for key in h5f2.keys():
                if key[:3] == 'L1_':
                    L1bits_labels.append(key)
                    L1bits.append(np.array(h5f2[key]))
                elif key == 'L1bit':
                    L1bit = np.array(h5f2[key])
                elif key == 'event_info':
                    # Event info names taken from here:
                    # https://gitlab.cern.ch/cms-l1-ad/l1tntuple-maker/-/blob/master/convert_to_h5.py#L125
                    #                     labels = ["run", "lumi", "event", "bx", "orbit", "time","nPV_True"] #deprecated
                    labels = ['run', 'lumi', 'event', 'bx', 'nPV', 'nPV_Good']
                    index = 0
                    for label in labels:
                        eventData[label] = h5f2[key][:, index]
                        index += 1

                if len(h5f2[key].shape) < 3:
                    continue
                if key == 'full_data_cyl':
                    data = h5f2[key][:, :, :].astype('float')
        else:
            if process not in h5f2.keys():
                raise ValueError(f'The process {process} is not contained in the file {inputfile}.')
            for key in h5f2.keys():
                if key.startswith(process + '_L1_'):
                    L1bits_labels.append(key.replace(process + '_', ''))
                    L1bits.append(np.array(h5f2[key]))
                elif key == process + '_l1bit':
                    L1bit = np.array(h5f2[key])

                # doing this should remove all trigger things, and leave a single entry with the data
                if len(h5f2[key].shape) < 3:
                    continue
                if key == process:
                    data = h5f2[key][:, :, :].astype('float')

    # splitting objects
    if object_ranges == 'default1':
        # we have 57 variables, but they do not have labels yet. Lets assign them based on the info in
        # https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae/-/blob/master/in/prep_data.py
        # I assume that we have MET, 4 electrons, 4 muons and 10 jets
        # These are 19 objects, times 3 parameters -> 57 vars
        # From line 27 I think the order is as I listed it: MET, egs, muons, jets
        object_ranges = {'met': 0, 'egs': (1, 5), 'muons': (5, 9), 'jets': (9, 19)}
        if verbosity > 0:
            print(
                'Using the object ranges that are consistent with old h5 files provided by the anomaly detection team containing MET, 4 electrons, 4 muons and 10 jets in this order'
            )
    elif object_ranges == 'default2':
        object_ranges = {'met': 0, 'egs': (1, 13), 'muons': (13, 21), 'jets': (21, 33)}
        if verbosity > 0:
            print(
                'Using the object ranges that are consistent with new h5 files provided by the anomaly detection team containing MET, 12 electrons, 8 muons and 12 jets in this order'
            )

    if verbosity > 0:
        print(f'Object ranges for reading the {process} dataset: {object_ranges}')

    np_sums = data[:, object_ranges['met'], :].reshape((data.shape[0], 1, 3))  # reshape is needed to keep dimensionality
    np_egs = data[:, object_ranges['egs'][0] : object_ranges['egs'][1]]
    np_muons = data[:, object_ranges['muons'][0] : object_ranges['muons'][1]]
    np_jets = data[:, object_ranges['jets'][0] : object_ranges['jets'][1]]

    # converting to awkward
    ak_egs = ak.zip(
        {key: ak.from_regular(np_egs[:, :, i], axis=1) for i, key in enumerate(['pt', 'eta', 'phi'])}, with_name='Momentum4D'
    )
    ak_muons = ak.zip(
        {key: ak.from_regular(np_muons[:, :, i], axis=1) for i, key in enumerate(['pt', 'eta', 'phi'])}, with_name='Momentum4D'
    )
    ak_jets = ak.zip(
        {key: ak.from_regular(np_jets[:, :, i], axis=1) for i, key in enumerate(['pt', 'eta', 'phi'])}, with_name='Momentum4D'
    )

    # energy sums are handled a bit differently
    ak_sums = ak.zip(
        {key: ak.from_regular(np_sums[:, :, 2 * i], axis=1) for i, key in enumerate(['pt', 'phi'])}, with_name='Momentum4D'
    )
    ak_sums['type'] = [2] * len(ak_sums)  # MET should have Type 2

    # removing empty entries (not needed for energy sums)
    ak_egs = ak_egs[ak_egs.pt > 0]
    ak_muons = ak_muons[ak_muons.pt > 0]
    ak_jets = ak_jets[ak_jets.pt > 0]

    infoDict['nEvents'] = len(ak_muons)

    # formating the L1 bits
    d_bits = dict(zip(L1bits_labels, np.asarray(L1bits)))
    d_bits['total L1'] = L1bit
    df_bits = ak.zip(d_bits)

    # after everything else: add moreInfo
    if moreInfo:
        infoDict = {**infoDict, **moreInfo}

    if verbosity > 0:
        print('Done!')

    # we'll output the data as a dict, makes it easier later
    dataDict = {}
    dataDict['muons'] = ak_muons
    dataDict['egs'] = ak_egs
    dataDict['jets'] = ak_jets
    dataDict['sums'] = ak_sums

    return infoDict, eventData, ak.Array(dataDict), df_bits


def readFromAnomalySignalh5(inputfile, process, object_ranges='default1', moreInfo=None, verbosity=0):
    # object ranges -- dictionary containing the keys "met", "jets", "muons" and "egs" and for each key a tuple containing
    #                  the range of the corresponding objects in the data array of the h5 file. For "met" it should be given
    #                  only the index (one number, no tuple), because by definition there can be only one MET object
    #                  Alternatively it can be set to the string "default1" (4 egs, 4 muons, 10 jets) or
    #                  "default2" (12 egs, 8 muons, 12 jets)

    return readFromAnomalyh5(inputfile, process, object_ranges, moreInfo, verbosity)


def readFromAnomalyBackgroundh5(inputfile, object_ranges='default1', moreInfo=None, verbosity=0):
    # object ranges -- dictionary containing the keys "met", "jets", "muons" and "egs" and for each key a tuple containing
    #                  the range of the corresponding objects in the data array of the h5 file. For "met" it should be given
    #                  only the index (one number, no tuple), because by definition there can be only one MET object
    #                  Alternatively it can be set to the string "default1" (4 egs, 4 muons, 10 jets) or
    #                  "default2" (12 egs, 8 muons, 12 jets)

    return readFromAnomalyh5(inputfile, 'background', object_ranges, moreInfo, verbosity)


def awkward_to_numpy(ak_array, maxN, verbosity=0):
    # this is a bit ugly, but it works. Maybe we can improve later
    selected_arr = ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), {'pt': 0, 'eta': 0, 'phi': 0})
    np_arr = np.stack((selected_arr.pt.to_numpy(), selected_arr.eta.to_numpy(), selected_arr.phi.to_numpy()), axis=2)
    return np_arr.reshape(np_arr.shape[0], np_arr.shape[1] * np_arr.shape[2])


def mutual_information_bernoulli_loss(y_true, y_pred, use_quantized_sigmoid=False, bits_bernoulli_sigmoid=8):
    """
    I(x;y)  = H(x)   - H(x|y)
            = H(L_n) - H(L_n|s)
            = H(L_n) - (H(L_n|s=0) + H(L_n|s=1))
    H_bernoulli(x) = -(1-theta) x ln(1-theta) - theta x ln(theta)
    here theta => probability for 1 and 1-theta => probability for 0

    pseudocode:
    def get_h_bernoulli(l):
        theta = np.mean(l, axis=0)
        return -(1-theta) * np.log2(1-theta) - theta * np.log2(theta)

    y_pred = np.random.binomial(n=1, p=0.6, size=[2000, 5])
    y_true = np.random.binomial(n=1, p=0.6, size=[2000])

    y_pred[y_true == 0] = np.random.binomial(n=1, p=0.5, size=[len(y_true[y_true == 0]), 5])
    y_pred[y_true == 1] = np.random.binomial(n=1, p=0.8, size=[len(y_true[y_true == 1]), 5])

    H_L_n = get_h_bernoulli(y_pred)
    H_L_n_s0 = get_h_bernoulli(y_pred[y_true == 0])
    H_L_n_s1 = get_h_bernoulli(y_pred[y_true == 1])

    counts = np.bincount(y_true)

    MI = H_L_n - ((counts[0] / 2000 * H_L_n_s0) + (counts[1] / 2000 * H_L_n_s1))

    return np.sum(MI)

    :param y_pred: output of the layer
    :param y_true: sensitive attribute
    :return: The loss
    """

    def get_h_bernoulli(tensor):
        def get_theta(x):
            alpha = None
            temperature = 6.0
            use_real_sigmoid = use_quantized_sigmoid
            # quantized sigmoid
            _sigmoid = quantized_sigmoid(bits=bits_bernoulli_sigmoid, use_stochastic_rounding=True, symmetric=True)
            if isinstance(alpha, str):
                assert alpha in ['auto', 'auto_po2']

            if isinstance(alpha, str):
                len_axis = len(x.shape)

                if len_axis > 1:
                    if keras.backend.image_data_format() == 'channels_last':
                        axis = list(range(len_axis - 1))
                    else:
                        axis = list(range(1, len_axis))
                else:
                    axis = [0]

                std = ops.std(x, axis=axis, keepdims=True) + keras.backend.epsilon()
            else:
                std = 1.0

            if use_real_sigmoid:
                p = ops.sigmoid(temperature * x / std)
            else:
                p = tf.cast(_sigmoid(temperature * x / std), tf.float64)

            return p

        def log2(x):
            numerator = tf.math.log(x + 1e-20)
            denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
            return numerator / denominator

        theta = tf.reduce_mean(get_theta(tensor), axis=0)

        return tf.reduce_sum(-(1 - theta) * log2(1 - theta) - theta * log2(theta))

    def compute_for_value(value, y_true, y_pred, get_h_bernoulli):
        if tf.shape(y_true).shape[0] == 1:
            y_filter = tf.where(y_true == value)
        else:
            y_filter = tf.where(y_true[:, 0] == value)[:, 0]

        y_i = tf.gather(y_pred, indices=y_filter)
        H_L_n_si = get_h_bernoulli(y_i)
        cnt_i = tf.shape(y_i)[0] + tf.cast(1e-16, dtype=tf.int32)  # number of repr with index i
        norm_si = cnt_i / tf.shape(y_pred)[0]
        return H_L_n_si, norm_si

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float64)
    H_L_n = get_h_bernoulli(y_pred)

    unique_y_true, _ = tf.unique(y_true)

    H_L_n_s = []
    norm_s = []

    def compute_per_value(v):
        return compute_for_value(v, y_true, y_pred, get_h_bernoulli)

    results = tf.map_fn(compute_per_value, unique_y_true, fn_output_signature=(tf.float64, tf.float64))

    H_L_n_s = tf.convert_to_tensor(results[0])
    norm_s = tf.convert_to_tensor(results[1])

    MI = H_L_n - tf.reduce_sum(tf.math.multiply(norm_s, H_L_n_s))

    # NOTE: this is a hotfix when we dont have all classes
    MI = tf.where(tf.math.is_nan(MI), tf.convert_to_tensor([0.0], dtype=tf.float64), MI)

    return tf.cast(MI, dtype=tf.float32)


class MILoss(keras.losses.Loss):
    def __init__(self, use_quantized_sigmoid=False, bits_bernoulli_sigmoid=8, name="mi_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_quantized_sigmoid = use_quantized_sigmoid
        self.bits_bernoulli_sigmoid = bits_bernoulli_sigmoid

    def call(self, y_true, y_pred):
        return mutual_information_bernoulli_loss(
            y_true,
            y_pred,
            self.use_quantized_sigmoid,
            self.bits_bernoulli_sigmoid
        )

    def get_config(self):
        return super().get_config()

# following code is based on: https://www.kaggle.com/code/matteomonzali/nn-model

def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = np.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = np.cumsum(np.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return np.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = np.array(predictions)
    masses = np.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[np.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = np.argsort(np.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return np.mean(cvms)


def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve

    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = np.concatenate([sample_weights_zero, sample_weights_one])
    data_all = np.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr


def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.

    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = np.array(data_prediction), np.array(mc_prediction)
    weights_data, weights_mc = np.array(weights_data), np.array(weights_mc)

    assert np.all(data_prediction >= 0.) and np.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert np.all(mc_prediction >= 0.) and np.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= np.sum(weights_data)
    weights_mc /= np.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = np.max(np.abs(fpr - tpr))
    return Dnm


def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.

    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert np.all(predictions >= 0.) and np.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = np.minimum(tpr, tpr_thresholds[index])
        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))
    tpr_thresholds = np.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= np.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * np.array(roc_weights))
    return area

def load_tau_data(path):

    if os.path.isfile(f"{path}/x.npy"):
        x = np.load(f"{path}/x.npy")
        y = np.load(f"{path}/y.npy")
        s = np.load(f"{path}/s.npy")
        test = pd.read_pickle(f"{path}/test")
        correlation = pd.read_pickle(f"{path}/correlation")
        agreement_test = pd.read_pickle(f"{path}/agreement_test")

        return x, y, s, test, correlation, agreement_test

    # load data
    # The label ‘signal’ being ‘1’ for signal events, ‘0’ for background events
    # the signal events are simulated, while background events are real data.
    training = pd.read_csv(f"{path}/training.csv")
    # The test dataset has all the columns that training.csv has, except
    # mass, production, min_ANNmuon, and signal.
    test = pd.read_csv(f"{path}/test.csv")
    # Simulated and real events from the channel Ds → φπ to evaluate
    # the performance for simulated-real data
    # It contains the same columns as test.csv and weight column. 
    agreement = pd.read_csv(f"{path}/check_agreement.csv")
    # Only real background events recorded at LHCb to evaluate
    # the mass values locally.
    # It contains the same columns as test.csv and mass column to check correlation
    correlation = pd.read_csv(f"{path}/check_correlation.csv")

    # simulation (source) (signal==1) contains 8205 examples 
    # real data (target) (signal==0) contains 322942 examples with weight values
    counts = np.unique(agreement["signal"], return_counts=True)
    counts_train = np.unique(training["signal"], return_counts=True)

    print(f"Agreement data # source domain: {counts[1][1]}, # target domain: {counts[1][0]}")
    print(f"Training data # source domain: {counts_train[1][1]}, # target domain: {counts_train[1][0]}")

    # prepare the data

    # split the data for agreement test at the end
    agreement_train, agreement_test = \
        train_test_split(agreement, test_size=0.666)

    # we take equal values from both groups
    agreement_train_s0 = agreement_train[agreement_train["signal"]==0].sample(n=int(len(agreement_train[agreement_train["signal"]==1])/2))
    agreement_train_s1 = agreement_train[agreement_train["signal"]==1].sample(n=int(len(agreement_train[agreement_train["signal"]==1])/2))
    agreement = pd.concat([agreement_train_s0, agreement_train_s1])

    # prepare the data for the domain adaptation task
    s = np.concatenate([agreement_train["signal"].to_numpy(), training["signal"].to_numpy()])
    agreement_train = agreement_train.drop(columns=["id", "signal", "SPDhits", "weight"], axis = 1)

    # from the agreement data we only have background so y=0
    y = np.concatenate([np.zeros(len(agreement_train)), training["signal"].to_numpy()])

    # Note: for this example we dont use the mass
    # mass_train = training["mass"]

    # create the features to train on
    x = training.drop(columns=["id", "production", "min_ANNmuon", "signal", "mass", "SPDhits"], axis = 1)
    x = pd.concat([agreement_train, x], ignore_index=True)

    np.save(f"{path}/x", x)
    np.save(f"{path}/y", y)
    np.save(f"{path}/s", s)
    test.to_pickle(f"{path}/test") 
    correlation.to_pickle(f"{path}/correlation") 
    agreement_test.to_pickle(f"{path}/agreement_test")

    return x, y, s, test, correlation, agreement_test


def load_tau_data_split(path):

    if os.path.isfile(f"{path}/X_train.npy"):
        X_train = np.load(f"{path}/X_train.npy")
        X_test = np.load(f"{path}/X_test.npy")
        y_train = np.load(f"{path}/y_train.npy")
        y_test = np.load(f"{path}/y_test.npy")
        S_train = np.load(f"{path}/S_train.npy")
        S_test = np.load(f"{path}/S_test.npy")
        agreement_weight = np.load(f"{path}/agreement_weight.npy")
        agreement_signal = np.load(f"{path}/agreement_signal.npy")
        agreement_test_feature = np.load(f"{path}/agreement_test_feature.npy")
        test = np.load(f"{path}/test.npy")
        correlation = np.load(f"{path}/correlation.npy")
        mass = np.load(f"{path}/mass.npy")
        scaler = load(f"{path}/scaler.joblib")

        train_vali_data = [X_train, X_test, y_train, y_test, S_train, S_test]
        agreement_data = [agreement_weight, agreement_signal, agreement_test_feature]
        meta_data = [test, correlation, scaler, mass]

        return train_vali_data, agreement_data, meta_data

    x, y, s, test, correlation, agreement_test = load_tau_data(path)

    # split the data into train and test set
    X_train, X_test, y_train, y_test, S_train, S_test = \
        train_test_split(x, y, s, test_size=0.333)

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # prepare agreement data
    agreement_weight = agreement_test["weight"]
    agreement_signal = agreement_test["signal"]
    agreement_test_feature = agreement_test.drop(columns=["id", "signal", "SPDhits", "weight"])
    agreement_test_feature = scaler.transform(agreement_test_feature)

    train_vali_data = [X_train, X_test, y_train, y_test, S_train, S_test]
    agreement_data = [agreement_weight, agreement_signal, agreement_test_feature]
    mass = correlation["mass"]
    correlation = correlation.drop(['id', 'mass', 'SPDhits'], axis=1)
    meta_data = [test, correlation, scaler, mass]

    np.save(f"{path}/X_train", X_train)
    np.save(f"{path}/X_test", X_test)
    np.save(f"{path}/y_train", y_train)
    np.save(f"{path}/y_test", y_test)
    np.save(f"{path}/S_train", S_train)
    np.save(f"{path}/S_test", S_test)
    np.save(f"{path}/agreement_weight", agreement_weight)
    np.save(f"{path}/agreement_signal", agreement_signal)
    np.save(f"{path}/agreement_test_feature", agreement_test_feature)
    np.save(f"{path}/test", test)
    np.save(f"{path}/correlation", correlation)
    np.save(f"{path}/mass", mass)
    dump(scaler, f"{path}/scaler.joblib")

    return train_vali_data, agreement_data, meta_data
