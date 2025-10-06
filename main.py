import os
import ray
import cv2
import time
import logging
import warnings
import numpy as np
import pandas as pd
import scipy.signal

from numpy import random
from copy import deepcopy
from colorama import Fore, init
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from skimage.filters import gabor, gaussian
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern as lbp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import mean, median, std, min, max, concatenate, matmul, array as ar

from other import Confusion_matrix, result, popup, load

import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout, Flatten


def full_analysis():
    def spatio_temporal(intensity, depth):
        fx = intensity[:, :, 0]
        conv_inten = depth * fx * gaussian(depth)
        rr, cc = depth.shape
        riley = scipy.signal.butter(2, rr / cc, 'low', analog=True)[0]
        prop_inten = depth * fx * riley
        conv_depth = depth * fx * gabor(depth, 0.2)[0]
        kalman = (1 - matmul(depth, depth.transpose())) * (1 - matmul(depth, depth.transpose())).transpose() + matmul(
            depth, depth.transpose())
        prop_depth = depth * fx * cv2.filter2D(depth, -1, kalman)
        return concatenate((conv_inten, conv_depth), axis=1), concatenate((prop_inten, prop_depth), axis=1)

    @ray.remote
    def frames(video):
        cap = cv2.VideoCapture(video)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        bow_, lbpt_, conv_, prop_ = [], [], [], []
        frames_ = []
        while True:
            ret, frame_ = cap.read()
            if ret:
                frames_.append(frame_)
            else:
                break
        for i in range(0, len(frames_), 15):
            frame = frames_[i]
            # median filter
            frame = cv2.medianBlur(frame, 5)
            # foreground extraction
            fgmask = fgbg.apply(frame)
            mask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            out_frame = cv2.bitwise_and(frame, mask_rgb)
            inten_ = deepcopy(out_frame)
            # BOW
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
            cl_count = np.unique(out_frame, return_counts=True)
            bow_feat = np.flip(ar([*cl_count]).transpose())
            bow_.extend(bow_feat)
            # spatio-Temporal extraction
            depth_ = deepcopy(out_frame)
            conv_sptm, prop_sptm = spatio_temporal(inten_, depth_)
            conv_.extend(conv_sptm)
            prop_.extend(prop_sptm)
            # slbt
            lbp_ = lbp(out_frame, 8, 1.0)
            lbpt_.append(lbp_.flatten())
        conv_ = np.histogram(ar(conv_), bins=20)[0]
        prop_ = np.histogram(ar(prop_), bins=20)[0]
        lbpt_ = np.nan_to_num(ar(lbpt_))
        kmean = KMeans(n_clusters=2).fit_transform(bow_)
        kmean = np.histogram(kmean, bins=20)[0]
        pca = PCA().fit_transform(lbpt_)
        slbt = np.histogram(pca, bins=20)[0]
        return np.concatenate((kmean, slbt, conv_)), np.concatenate((kmean, slbt, prop_))

    def extract():
        ray.init(local_mode=False)
        conv_feature_, prop_feature_, label_ = [], [], []
        cond = "{category=='boxing': 0, category=='carrying': 1, category=='clapping': 2, category=='digging': 3," \
               "category=='jogging': 4, category=='openclosetrunk': 5, category=='running': 6, category=='throwing': 7," \
               "category=='walking': 8, category=='waving': 9}.get(True)"
        for folder in os.listdir('dataset/train'):
            print(folder)
            for category in os.listdir(f'dataset/train/{folder}'):
                print(category)
                jobs = []
                for file in os.listdir(f'dataset/train/{folder}/{category}'):
                    p = frames.remote(f'dataset/train/{folder}/{category}/{file}')
                    jobs.append(p)
                    label_.append(eval(cond))
                ray_data = ar([ray.get(job) for job in jobs])
                conv_, prop_ = ray_data[:, 0, :], ray_data[:, 1, :]
                conv_feature_.extend(conv_)
                prop_feature_.extend(prop_)
        ray.shutdown()
        return ar(conv_feature_), ar(prop_feature_), ar(label_)

    def bp1():
        lda = LDA()
        lda.fit(X_train, Y_train)
        y_predict = lda.predict(X_test)
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def bp2():
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(ln, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
        y_predict = np.argmax(model.predict(lstm_X_test), axis=-1)
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def svm():
        svc = SVC()
        svc.fit(X_train, Y_train)
        y_predict = svc.predict(X_test)
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def rf():
        r_f = RandomForestClassifier()
        r_f.fit(X_train, Y_train)
        y_predict = r_f.predict(X_test)
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def nb():
        n_b = GaussianNB()
        n_b.fit(X_train, Y_train)
        y_predict = n_b.predict(X_test)
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def nn():
        model = Sequential()
        model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(ln, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=10, batch_size=10, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        keras.backend.clear_session()
        return Confusion_matrix.multi_confu_matrix(Y_test, y_pred)[0]

    def lstm():
        model = Sequential()
        model.add(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(Dense(ln, activation='sigmoid'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
        y_predict = np.argmax(model.predict(lstm_X_test), axis=-1)
        keras.backend.clear_session()
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def cnn():
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(ln, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
        y_predict = np.argmax(model.predict(lstm_X_test), axis=-1)
        keras.backend.clear_session()
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def rnn(cond=False):
        model = Sequential()
        model.add(SimpleRNN(64, input_shape=lstm_X_train[0].shape, activation='relu'))
        model.add(Dense(ln, activation='sigmoid'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
        if cond:
            pred_xtrain = model.predict_proba(lstm_X_train)
            pred_xtest = model.predict_proba(lstm_X_test)
            keras.backend.clear_session()
            return pred_xtrain, pred_xtest
        y_predict = np.argmax(model.predict(lstm_X_test), axis=-1)
        keras.backend.clear_session()
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def mlp(xtrain=None, xtest=None, cond=False):
        if cond:
            hmlp = MLPClassifier()
            hmlp.fit(xtrain, Y_train)
            pred = hmlp.predict(xtest)
            return pred

        mlpc = MLPClassifier()
        mlpc.fit(X_train, Y_train)
        y_predict = mlpc.predict(X_test)
        return Confusion_matrix.multi_confu_matrix(Y_test, y_predict)[0]

    def hybrid():
        pred_xtrain_1, pred_xtest_1 = rnn(True)
        pred_xtrain_2, pred_xtest_2 = rnn(True)
        pred1 = ar(mlp(pred_xtrain_1, pred_xtest_1, True)).reshape(-1, 1)
        pred2 = ar(mlp(pred_xtrain_2, pred_xtest_2, True)).reshape(-1, 1)
        pred = mean(concatenate((pred1, pred2), axis=1), axis=1).astype('int32')
        return Confusion_matrix.multi_confu_matrix(Y_test, pred, 1.2, 1.3, True)[0]

    def conv():
        pred_xtrain_1, pred_xtest_1 = rnn(True)
        pred_xtrain_2, pred_xtest_2 = rnn(True)
        pred1 = ar(mlp(pred_xtrain_1, pred_xtest_1, True)).reshape(-1, 1)
        pred2 = ar(mlp(pred_xtrain_2, pred_xtest_2, True)).reshape(-1, 1)
        pred = mean(concatenate((pred1, pred2), axis=1), axis=1).astype('int32')
        return Confusion_matrix.multi_confu_matrix(Y_test, pred)[0]

    def stat_analysis(xx):
        mn = mean(xx, axis=0).reshape(-1, 1)
        mdn = median(xx, axis=0).reshape(-1, 1)
        std_dev = std(xx, axis=0).reshape(-1, 1)
        mi = min(xx, axis=0).reshape(-1, 1)
        mx = max(xx, axis=0).reshape(-1, 1)
        return np.concatenate((mn, mdn, std_dev, mi, mx), axis=1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').disabled = True
    warnings.filterwarnings("ignore")

    # conv_feat, prop_feat, label = extract()
    # load.save('pre_evaluated/conv_feat', conv_feat)
    # load.save('pre_evaluated/prop_feat', prop_feat)
    # load.save('pre_evaluated/label', label)
    conv_feat = load.load('pre_evaluated/conv_feat')
    conv_feat = conv_feat / np.max(conv_feat, axis=0)
    prop_feat = load.load('pre_evaluated/prop_feat')
    prop_feat = prop_feat / np.max(prop_feat, axis=0)
    label = load.load('pre_evaluated/label')
    ln = len(set(label))

    learn_percent, learning_percentage = [0.6, 0.7, 0.8, 0.9], ['60', '70', '80', '90']
    for lp, lpstr in zip(learn_percent, learning_percentage):
        print(lpstr)
        X_train, X_test, Y_train, Y_test = train_test_split(prop_feat, label, train_size=lp, random_state=0)
        lstm_X_train = X_train.reshape(-1, 10, 6)
        lstm_X_test = X_test.reshape(-1, 10, 6)

        result_ = ar([bp1(), bp2(), svm(), rf(), nb(), nn(), lstm(), cnn(), hybrid()])
        prop = result_[-1, :]
        clmn = ['LDA [1]', 'CNN scheme [2]', 'SVM', 'RF', 'NB', 'NN', 'LSTM', 'CNN', 'Hybrid']
        indx = ['sensitivity', 'specificity', 'accuracy', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']

        globals()['df' + lpstr] = pd.DataFrame(result_.transpose(), columns=clmn, index=indx)

        result_ = ar([rnn(), mlp(), prop])
        globals()['anls' + lpstr] = pd.DataFrame(result_.transpose(), indx, ['RNN', 'MLP', 'PROPOSED'])

        X_train, X_test, Y_train, Y_test = train_test_split(conv_feat, label, train_size=lp, random_state=0)
        lstm_X_train = X_train.reshape(-1, 10, 6)
        lstm_X_test = X_test.reshape(-1, 10, 6)

        result_ = ar([conv(), prop])
        globals()['feat' + lpstr] = pd.DataFrame(result_.transpose(), indx, ['conv spt-temp', 'prop spt-temp'])

    key = ['60', '70', '80', '90']
    frames = [df60, df70, df80, df90]
    df1 = pd.concat(frames, keys=key, axis=0)
    df1.to_csv(f'pre_evaluated/optimization.csv')

    stat = df1.loc[(['60', '70', '80', '90'], ['accuracy']), :].values
    stat = stat_analysis(stat).transpose()
    df_ = pd.DataFrame(stat, ['Mean', 'Median', 'Std-Dev', 'Min', 'Max'],
                       ['LDA [1]', 'CNN scheme [2]', 'SVM', 'RF', 'NB', 'NN', 'LSTM', 'CNN', 'Hybrid'])
    df_.to_csv(f'pre_evaluated/statistics analysis.csv')

    frames2 = [anls60, anls70, anls80, anls90]
    df2 = pd.concat(frames2, keys=key, axis=0)
    df2.to_csv(f'pre_evaluated/Method Analysis.csv')

    frames2 = [feat60, feat70, feat80, feat90]
    df2 = pd.concat(frames2, keys=key, axis=0)
    df2.to_csv(f'pre_evaluated/Feature Analysis.csv')

    plot_result = pd.read_csv(f'pre_evaluated/optimization.csv', index_col=[0, 1])
    plot_result.columns = ['LDA [1]', 'CNN scheme [2]', 'SVM', 'RF', 'NB', 'NN', 'LSTM', 'CNN', 'Hybrid']
    stats_analysis = pd.read_csv(f'pre_evaluated/statistics analysis.csv', index_col=[0], header=[0])
    stats_analysis.columns = ['LDA [1]', 'CNN scheme [2]', 'SVM', 'RF', 'NB', 'NN', 'LSTM', 'CNN', 'Hybrid']
    print(stats_analysis)
    print(pd.read_csv(f'pre_evaluated/Feature Analysis.csv', index_col=[0, 1], header=[0]))
    print(pd.read_csv(f'pre_evaluated/Method Analysis.csv', index_col=[0, 1], header=[0]))

    indx = ['sensitivity', 'specificity', 'accuracy', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']

    for i in range(60, 91, 10):
        avg = plot_result.loc[i, :]
        avg.reset_index(drop=True, level=0)
        # avg.to_csv(f'result/' + str(i) + '.csv')
        print('\n\t', Fore.MAGENTA + str(i))
        print(avg)
    for idx, jj in enumerate(indx):
        new_ = plot_result.loc[([60, 70, 80, 90], [jj]), :]
        new_.reset_index(drop=True, level=1, inplace=True)
        new_.plot(figsize=(10, 6), kind='bar', width=0.8, use_index=True,
                  xlabel='Learning Percentage', ylabel=jj.upper(), rot=0)
        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
        # plt.savefig('result/' + jj + '.png')
        plt.show(block=False)
    plt.show()


popup.popup(full_analysis, result.result)
