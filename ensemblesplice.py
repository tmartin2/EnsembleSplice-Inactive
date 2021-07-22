from Data import encode_data
from Models import sub_models
from Models import build_models
from Visuals import make_report
# from utils import cross_validation
# from utils import avg_loss_acc_by_fold_esplice
from Models import utils
from sklearn.linear_model import LogisticRegression# ensemble method
from sklearn.linear_model import Perceptron # ensemble method
from sklearn.svm import LinearSVC # ensemble method
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # cross validation
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt # plotting
import matplotlib.font_manager as font_manager # plot fonts
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import copy

# class EnsembleSplice(tf.keras.Model):

def build_ensemble(
        hmd,
        ensemble_type,
        batch_size,
        kwargs
    ):
    # [history 0 , model 1 , X_trn 2 , y_trn 3 , X_val 4 , y_val 5] * sub_model_iter count,
    # extraction function for getting labels from to_categorical encoding
    # For donor, 0 (false donor) -> [1 0], 1 (true donor) -> [0 1]
    # Pred looks like [9.5078093e-01 4.9219094e-02] for [1 0] means False donor
    # USE NP AMAX, axis=1 then add noise to predictions

    # function to convert [9.5078093e-01 4.9219094e-02] to 0
    extr = lambda vals: np.argmax(vals, axis=1)

    if kwargs['validate']:

        # HMD (dnn1_1 : [(history, model, X_val, y_val, X_trn, y_trn), (), (), (), ..., num_folds])

        # get the labels right. network 1 fold 1,2,3,4... labels
        val_predictions = dict([(f'Fold_{fold}', []) for fold in range(1, kwargs['folds']+1)])

        for sub_arch in hmd.keys():
            for index, model_by_fold in enumerate(hmd[sub_arch]):
                val_predictions[f'Fold_{index+1}'].append(extr(model_by_fold['model'].predict(model_by_fold['X_val'].astype('float32'), batch_size=batch_size)).reshape(-1,1))

        for fold_preds in val_predictions.keys():
            # note that they have the same keys
            val_predictions[fold_preds] = np.concatenate(val_predictions[fold_preds], axis=1)


        m = kwargs['models']

        # they're equal!!!
        # for index, x in enumerate(hmd[f'{m[0][0]}_1'][1]['y_val']):
        #     assert x.tolist()[0]==hmd[f'{m[1][0]}_1'][1]['y_val'][index].tolist()[0],'poop'

        if ensemble_type == 'esplice_l':
            val_scores_by_fold = []
            for index, fold_preds in enumerate(val_predictions.keys()):
                classifier = LogisticRegression(random_state=None).fit(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_trn']))
                clf_val_score = classifier.score(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_val']))
                val_scores_by_fold.append(clf_val_score)
            return np.mean(np.asarray(val_scores_by_fold))

        if ensemble_type == 'esplice_p':
            val_scores_by_fold = []
            for index, fold_preds in enumerate(val_predictions.keys()):
                classifier = Perceptron(shuffle=False, random_state=None).fit(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_trn']))
                clf_val_score = classifier.score(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_val']))
                val_scores_by_fold.append(clf_val_score)
            return np.mean(np.asarray(val_scores_by_fold))

        if ensemble_type == 'esplice_s':
            val_scores_by_fold = []
            for index, fold_preds in enumerate(val_predictions.keys()):
                # linear support vector classifier on training data
                classifier = LinearSVC(random_state=None).fit(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_trn']))
                # score on the validation data
                clf_val_score = classifier.score(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_val']))
                val_scores_by_fold.append(clf_val_score)
            return np.mean(np.asarray(val_scores_by_fold))

    if kwargs['train']:

        # HMD (dnn1_1 : {history, model, X_val, y_val, X_trn, y_trn })

        trn_predictions = []
        for sub_arch in hmd.keys():
            trn_predictions.append(extr(hmd[sub_arch]['model'].predict(hmd[sub_arch]['X_trn'].astype('float32'), batch_size=batch_size)).reshape(-1,1))
        trn_predictions = np.concatenate(trn_predictions, axis=1)

        m = kwargs['models']

        # GET ANOVA IN HERE
        # if kwargs['anova']:
        #     x = f_classif(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
        #     print('p-values',x.p_values)

        if ensemble_type == 'esplice_l':
            # logistic regressor on training data
            classifier = LogisticRegression(random_state=None).fit(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
            # score on the validation data
            clf_val_score = classifier.score(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
            return clf_val_score

        if ensemble_type == 'esplice_p':
            # perceptron on training data
            classifier = Perceptron(random_state=None).fit(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
            # score on the validation data
            clf_val_score = classifier.score(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
            return clf_val_score

        if ensemble_type == 'esplice_s':
            # linear support vector classifier on training data
            classifier = LinearSVC(random_state=None).fit(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
            # score on the validation data
            clf_val_score = classifier.score(trn_predictions, extr(hmd[f'{m[0][0]}_1']['y_trn']))
            return clf_val_score

    if kwargs['test']:

        # HMD (dnn1_1 : {model, X_test, y_test})

        tst_predictions = []
        for sub_arch in hmd.keys():
            tst_predictions.append(extr(hmd[sub_arch]['model'].predict(hmd[sub_arch]['X_test'].astype('float32'), batch_size=batch_size)).reshape(-1,1))
        tst_predictions = np.concatenate(tst_predictions, axis=1)

        m = kwargs['models']

        if ensemble_type == 'esplice_l':
            # logistic regressor on training data
            classifier = LogisticRegression(random_state=None).fit(tst_predictions, extr(hmd[f'{m[0][0]}_1']['y_test']))
            # score on the validation data
            clf_val_score = classifier.score(tst_predictions, extr(hmd[f'{m[0][0]}_1']['y_test']))
            return clf_val_score

        if ensemble_type == 'esplice_p':
            # perceptron on training data
            classifier = Perceptron(random_state=None).fit(tst_predictions, extr(hmd[f'{m[0][0]}_1']['y_test']))
            # score on the validation data
            clf_val_score = classifier.score(tst_predictions, extr(hmd[f'{m[0][0]}_1']['y_test']))
            return clf_val_score

        if ensemble_type == 'esplice_s':
            # linear support vector classifier on training data
            classifier = LinearSVC(random_state=None).fit(tst_predictions, extr(hmd[f'{m[0][0]}_1']['y_test']))
            # score on the validation data
            clf_val_score = classifier.score(tst_predictions, extr(hmd[f'{m[0][0]}_1']['y_test']))
            return clf_val_score


        # if ensemble_type == 'esplice_p':
        #     val_scores_by_fold = []
        #     for index, fold_preds in enumerate(trn_predictions.keys()):
        #         # Perceptron on training data
        #         classifier = Perceptron(shuffle=False, random_state=None).fit(trn_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_trn']))
        #         # score on the validation data
        #         clf_val_score = classifier.score(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_val']))
        #         val_scores_by_fold.append(clf_val_score)
        #     return np.mean(np.asarray(val_scores_by_fold))
        #
        # if ensemble_type == 'esplice_s':
        #     val_scores_by_fold = []
        #     for index, fold_preds in enumerate(trn_predictions.keys()):
        #         # linear support vector classifier on training data
        #         classifier = LinearSVC(random_state=None).fit(trn_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_trn']))
        #         # score on the validation data
        #         clf_val_score = classifier.score(val_predictions[fold_preds], extr(hmd[f'{m[0][0]}_1'][index]['y_val']))
        #         val_scores_by_fold.append(clf_val_score)
        #     return np.mean(np.asarray(val_scores_by_fold))



            # # fit perceptron on training data
            # clf01 = Perceptron(shuffle=False, random_state=None).fit(all_trn_preds, trn_labels)
            # # score on the validation data
            # clf01_val_scores = clf01.score(all_val_preds, val_labels)



        # # logistic regressor on training data
        # clf03 = LinearSVC(random_state=None).fit(all_trn_preds, trn_labels)
        # # score on the validation data
        # clf03_val_scores = clf03.score(all_val_preds, val_labels)



    # val_pred_dataset = []
    # trn_pred_dataset = []
    # for sub_arch in hmd.keys():
    #     hmd[sub_arch].predict()
    #
    #     extr(hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size)).reshape(-1,1))


    #val_labels = extr(hmd[0][5])
    #trn_labels = extr(hmd[0][3])

    # # note that all labels are the same given the same random state in encode_data
    # for hmd_tup in hmd:
    #     print(hmd_tup[3][:10])

    # check trn and val labels
    # for hmd_tup in hmd:
    #     assert hmd[0][3] == hmd_tup[3],'Trn Label sets are different'
    #     assert hmd[0][5] == hmd_tup[5],'Val Label sets are different'

    # thus, the selection of labels can be arbitrary

    # print(hmd[0][3])
    # print("TRAIN LABELS", trn_labels)
    #
    # # train data for classifier
    # trn_predictions = []
    # for hmd_tup in hmd:
    #     # show X_trn data
    #     #print(hmd_tup[2])
    #     # show y_trn data
    #     #print(hmd_tup[3])
    #     # show predictions
    #     print('TRN PREDICTIONS',hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size).shape)
    #     trn_predictions.append(extr(hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size)).reshape(-1,1))
    #     # reformat prediction to be concatenate-able
    #     #print('TRN CONVERTED PREDICTIONS',extr(hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size)).reshape(-1,1))
    #     #trn_predictions.append(extr(hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size)).reshape(-1,1))
    #
    # all_trn_preds = np.concatenate(trn_predictions, axis=1)
    # print('ALL TRAIN TOGETER', all_trn_preds)
    #
    # # validation data for classifiers
    # val_predictions = []
    # for hmd_tup in hmd:
    #     val_predictions.append(extr(hmd_tup[1].predict(hmd_tup[4].astype('float32'), batch_size=batch_size)).reshape(-1,1))
    # all_val_preds = np.concatenate(val_predictions, axis=1)
    #
    # # fit perceptron on training data
    # clf01 = Perceptron(shuffle=False, random_state=None).fit(all_trn_preds, trn_labels)
    # # score on the validation data
    # clf01_val_scores = clf01.score(all_val_preds, val_labels)
    #
    # # logistic regressor on training data
    # clf02 = LogisticRegression(random_state=None).fit(all_trn_preds, trn_labels)
    # # score on the validation data
    # clf02_val_scores = clf02.score(all_val_preds, val_labels)
    #
    # # logistic regressor on training data
    # clf03 = LinearSVC(random_state=None).fit(all_trn_preds, trn_labels)
    # # score on the validation data
    # clf03_val_scores = clf03.score(all_val_preds, val_labels)
    #
    #
    # # DNN
    # all_trn_preds = all_trn_preds.reshape(all_trn_preds.shape[0], 1, len(hmd))
    # all_val_preds = all_val_preds.reshape(all_val_preds.shape[0], 1, len(hmd))
    #
    # # for hmd_tup in hmd:
    # #     trn_predictions.append(hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size).reshape(-1,1))
    # # all_trn_preds = np.concatenate(trn_predictions, axis=1)
    #
    # # preds = hmd_tup[1].predict(hmd_tup[2].astype('float32'), batch_size=batch_size)
    # # all_trn_preds = np.reshape(preds.shape[0], 1, len(hmd))
    # # print(all_trn_preds.shape)
    # # print(all_trn_preds)
    #
    #
    # #print(all_val_preds.shape)
    # # noise injection
    # #all_trn_preds = np.random.normal(loc=all_trn_preds, scale=0.05)
    # #print(all_val_preds.shape)
    # # (2888, 1, # submodels)
    #
    #
    #
    # model = tf.keras.Sequential()
    # # 2888 1 sub_models
    # input_shape = (1, len(hmd))
    # # 2888 1 sub_models 2
    # #input_shape = (1, len(hmd), 2)
    # model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    # model.add(tf.keras.layers.Flatten())
    # #model.add(tf.keras.layers.GaussianNoise(0.5))
    # model.add(tf.keras.layers.Dense(
    #     units=2,
    #     activation='softmax',
    #     )
    # )
    # model.compile(
    #     loss='binary_crossentropy',
    #     optimizer='adam',
    #     metrics=[
    #         'accuracy',
    #         tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
    #         tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
    #         tf.keras.metrics.AUC(name='auc'),
    #         tf.keras.metrics.Precision(name='pre'),
    #         tf.keras.metrics.Recall(name='rec')
    #     ]
    # )
    # # print(all_trn_preds)
    # # print(hmd[0][3])
    # history = model.fit(
    #     x=all_trn_preds,
    #     y=hmd[0][3],
    #     batch_size=32,
    #     epochs=15, #14,
    # )
    #
    # score = model.evaluate(
    #     x=all_val_preds,
    #     y=hmd[0][5],
    #     batch_size=32,
    # )
    # print('MODEL LOSS', score[0])
    # print('MODEL ACC', score[1])
    #
    #
    # print((clf01_val_scores, clf02_val_scores, clf03_val_scores))
    #
    # return (clf01_val_scores, clf02_val_scores, clf03_val_scores)



        # # # ?not necessary
        # # val_predictions = []
        # # for hmd_tup in hmd:
        # #     trn_predictions.append(extr(hmd[1].predict(hmd[2].astype('float32'), batch_size=batch_size)).reshape(-1,1))
        # # all_trn_preds = np.concatenate(predictions, axis=1)
        #
        # # training
        # cnn_trn_pred = models[0].predict(cnn_X_trn.astype('float32'), batch_size=batch_size)
        # dnn_trn_pred = models[1].predict(rd_X_trn.astype('float32'), batch_size=batch_size)
        # rnn_trn_pred = models[2].predict(rd_X_trn.astype('float32'), batch_size=batch_size)
        # all_trn_preds = np.concatenate((extr(cnn_trn_pred).reshape(-1,1), extr(dnn_trn_pred).reshape(-1,1)), axis=1)#, extr(rnn_trn_pred).reshape(-1,1)
        #
        # # validation
        # cnn_val_pred = models[0].predict(cnn_X_val.astype('float32'), batch_size=batch_size)
        # dnn_val_pred = models[1].predict(rd_X_val.astype('float32'), batch_size=batch_size)
        # #rnn_val_pred = models[2].predict(rd_X_val.astype('float32'), batch_size=batch_size)
        # all_val_preds = np.concatenate((extr(cnn_val_pred).reshape(-1,1), extr(dnn_val_pred).reshape(-1,1)), axis=1)#, extr(rnn_val_pred).reshape(-1,1)
        #
        #
        # clf01 = Perceptron(shuffle=False).fit(all_trn_preds, extr(rd_y_trn))
        # val_scores = clf01.score(all_val_preds, extr(rd_y_val))
        #
        # clf02 = LogisticRegression(shuffle=False).fit(all_trn_preds, extr(rd_y_trn))
        # val_scores = clf02.score(all_val_preds, extr(rd_y_val))

        #return (trn_scores, val_scores)

def run(**kwargs):

    # dictionaries that are used to create one results dictionary for training, validation, and testing
    metrics = {
        'val_acc':[],'val_loss':[],'trn_acc':[],'trn_loss':[],
        'auc':[],'pre':[],'rec':[],'seATsp':[],'spATsn':[],
    }
    datasets = dict([(dataset, {}) for dataset in kwargs['datasets']])
    sub_models = [(model[0], dict([(num, {}) for num in range(1, model[1]+1)])) for model in kwargs['models']]
    esplicing = [(e_arch, {1:{}}) for e_arch in kwargs['esplicers']]
    models = dict(sub_models+esplicing)

    # a results dictionary acceptor: cnn1: 1: nn269: metrics, 2: nn269: metrics etc..
    results = {}
    for ss_type in kwargs['splice_sites']:
        results[ss_type] = copy.deepcopy(models)
        for model in results[ss_type].keys():
            for iter in results[ss_type][model].keys():
                results[ss_type][model][iter] = copy.deepcopy(datasets)
                for dataset in results[ss_type][model][iter].keys():
                    results[ss_type][model][iter][dataset] = copy.deepcopy(metrics)

    # encode the data based on the datasets and networks
    # encode_data should be run ONCE per dataset
    models_used = dict([(model[0], '') for model in kwargs['models']])
    encoded_data = {}
    for dataset in kwargs['datasets']:
        encoded_data[dataset] = models_used
        conv2d_dict = encode_data.encode(dataset, 'cnn1', kwargs) # could also be cnn5
        other_dict = encode_data.encode(dataset, 'cnn2', kwargs) # could also be not cnn1/cnn5
        for model in encoded_data[dataset].keys():
            if (model == 'cnn1') or (model=='cnn5'):
                encoded_data[dataset][model] = copy.deepcopy(conv2d_dict)
            else:
                encoded_data[dataset][model] = copy.deepcopy(other_dict)

    # perform k fold cross validation on the model archs. to evaluate model performance
    if kwargs['validate']:
        # find average of epochs
        average_best_vals = lambda val_list: np.mean(np.asarray([max(x) for x in val_list]))

        for ss_type in kwargs['splice_sites']:
            histories_models_data = dict([(f'{model[0]}_{iter}', []) for model in kwargs['models'] for iter in range(1, model[1]+1)])
            for model in models_used:
                for iter in results[ss_type][model].keys():
                    for dataset in kwargs['datasets']:
                        # this loop inspired by https://www.machinecurve.com/
                        #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
                        k_fold = KFold(n_splits=kwargs['folds'], shuffle=False, random_state=None)
                        fold_num = 1
                        folds = kwargs['folds']
                        X_train, y_train = encoded_data[dataset][model][ss_type]
                        for train, test in k_fold.split(X_train, y_train):
                            print()
                            print('#'*60)
                            print(f'\nTraining {model.upper()} instance {iter} for fold ({fold_num}/{folds}) on {ss_type.upper()} {dataset.upper()} data...\n')
                            print('#'*60)
                            print()
                            X_trn, y_trn = X_train[train], y_train[train]
                            X_val, y_val = X_train[test], y_train[test]
                            history, trained_model = build_models.build_single_model(
                                kwargs['rows'][ss_type][dataset],
                                ss_type,model,
                                X_trn,y_trn,X_val,y_val,
                                kwargs,
                            )
                            results[ss_type][model][iter][dataset]['val_acc'].append(history.history['val_accuracy'])
                            results[ss_type][model][iter][dataset]['val_loss'].append(history.history['val_loss'])
                            results[ss_type][model][iter][dataset]['trn_acc'].append(history.history['accuracy'])
                            results[ss_type][model][iter][dataset]['trn_loss'].append(history.history['loss'])
                            results[ss_type][model][iter][dataset]['auc'].append(history.history['val_auc'])
                            results[ss_type][model][iter][dataset]['pre'].append(history.history['val_pre'])
                            results[ss_type][model][iter][dataset]['rec'].append(history.history['val_rec'])
                            histories_models_data[f'{model}_{iter}'].append(
                                {
                                    'history':history, 'model':trained_model,
                                    'X_trn':X_trn, 'y_trn':y_trn, 'X_val':X_val, 'y_val':y_val,
                                }
                            )
                            fold_num += 1

                        # average out metric across epochs
                        results[ss_type][model][iter][dataset]['val_acc'] = average_best_vals(results[ss_type][model][iter][dataset]['val_acc'])
                        results[ss_type][model][iter][dataset]['val_loss'] = average_best_vals(results[ss_type][model][iter][dataset]['val_loss'])
                        results[ss_type][model][iter][dataset]['trn_acc'] = average_best_vals(results[ss_type][model][iter][dataset]['trn_acc'])
                        results[ss_type][model][iter][dataset]['trn_loss'] = average_best_vals(results[ss_type][model][iter][dataset]['trn_loss'])
                        results[ss_type][model][iter][dataset]['auc'] = average_best_vals(results[ss_type][model][iter][dataset]['auc'])
                        results[ss_type][model][iter][dataset]['pre'] = average_best_vals(results[ss_type][model][iter][dataset]['pre'])
                        results[ss_type][model][iter][dataset]['rec'] = average_best_vals(results[ss_type][model][iter][dataset]['rec'])

            # get results for each chosen ensemble_type
            for e_arch in kwargs['esplicers']:
                for dataset in kwargs['datasets']:
                    results[ss_type][e_arch][1][dataset] = build_ensemble(
                        histories_models_data,
                        e_arch,
                        32,
                        kwargs
                    )

    if kwargs['train']:
        best_vals = lambda val_list: max(val_list)
        for ss_type in kwargs['splice_sites']:
            histories_models_data = dict([(f'{model[0]}_{iter}', '') for model in kwargs['models'] for iter in range(1, model[1]+1)])
            for model in models_used:
                for iter in results[ss_type][model].keys():
                    for dataset in kwargs['datasets']:
                        X_train, y_train = encoded_data[dataset][model][ss_type]
                        print()
                        print('#'*60)
                        print(f'\nTraining {model.upper()} instance {iter} on {ss_type.upper()} {dataset.upper()} data...\n')
                        print('#'*60)
                        print()
                        history, trained_model = build_models.build_single_model(
                            kwargs['rows'][ss_type][dataset],
                            ss_type,model,
                            X_train,y_train,'','',
                            kwargs,
                        )

                        trained_model.save(f'./TrainedModels/{model.upper()}_Instance_{iter}_{dataset}_{ss_type}.h5')

                        results[ss_type][model][iter][dataset]['trn_acc'] = history.history['accuracy']
                        results[ss_type][model][iter][dataset]['trn_loss'] = history.history['loss']
                        histories_models_data[f'{model}_{iter}'] = {
                            'history':history, 'model':trained_model,
                            'X_trn':X_train, 'y_trn':y_train, 'X_val':'', 'y_val':'',
                        }

                        # average out metric across epochs
                        results[ss_type][model][iter][dataset]['trn_acc'] = best_vals(results[ss_type][model][iter][dataset]['trn_acc'])
                        results[ss_type][model][iter][dataset]['trn_loss'] = best_vals(results[ss_type][model][iter][dataset]['trn_loss'])

            # get results for each chosen ensemble_type
            for e_arch in kwargs['esplicers']:
                for dataset in kwargs['datasets']:
                    results[ss_type][e_arch][1][dataset] = build_ensemble(
                        histories_models_data,
                        e_arch,
                        32,
                        kwargs
                    )

    if kwargs['test']:
        best_vals = lambda val_list: max(val_list)
        for ss_type in kwargs['splice_sites']:
            histories_models_data = dict([(f'{model[0]}_{iter}', '') for model in kwargs['models'] for iter in range(1, model[1]+1)])
            for model in models_used:
                for iter in results[ss_type][model].keys():
                    for dataset in kwargs['datasets']:
                        X_test, y_test = encoded_data[dataset][model][ss_type]

                        train_model = build_models.build_single_model(
                            kwargs['rows'][ss_type][dataset],
                            ss_type, model,
                            '','','','',
                            kwargs,
                        )

                        train_model.load_weights(
                            os.getcwd()+f'/Models/TrainedModels/{model.upper()}_Instance_{iter}_{dataset}_{ss_type}.h5'
                        )

                        histories_models_data[f'{model}_{iter}'] = {
                            'model':train_model,
                            'X_test':X_test, 'y_test':y_test,
                        }

                        # score = trained_model.evaluate(
                        #     x=X_test,
                        #     y=y_test,
                        #     batch_size=32,
                        # )
                        # print(score)
                        # # average out metric across epochs
                        # results[ss_type][model][iter][dataset]['trn_acc'] = best_vals(results[ss_type][model][iter][dataset]['trn_acc'])
                        # results[ss_type][model][iter][dataset]['trn_loss'] = best_vals(results[ss_type][model][iter][dataset]['trn_loss'])

            # get results for each chosen ensemble_type
            for e_arch in kwargs['esplicers']:
                for dataset in kwargs['datasets']:
                    results[ss_type][e_arch][1][dataset] = build_ensemble(
                        histories_models_data,
                        e_arch,
                        32,
                        kwargs
                    )
    print(results)






        # # for each dataset
        # for dataset in datasets:
        #     encoded_data = dict([(s_model, '') for s_model in sub_models]) # encode data for each sub_model
        #     # for each sub_model
        #     for sub_model in sub_models:
        #         # encode the data
        #         encoded_data[sub_model] = encode_data.encode(dataset, sub_model, bal, test)
        #     for ss_type in splice_sites:
        #         if ss_type == 'acceptor':
        #             # index [:2] gets (acc_x, acc_y) of (acc_x, acc_y, don_x, don_y)
        #             X_train, y_train = list(encoded_data.values())[0][:2]
        #         if ss_type == 'donor':
        #             # which encoded values chosen here are arbitrary,
        #             # they're just used for the split
        #             X_train, y_train = list(encoded_data.values())[0][2:]
        #
        #         # this loop inspired by https://www.machinecurve.com/
        #         #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
        #         k_fold = KFold(n_splits=num_folds, shuffle=False)
        #         fold_num = 1
        #         # for train, test split of k folds
        #         for train, test in k_fold.split(X_train, y_train):
        #             # for each sub_model
        #             histories_models_data = []
        #             for sub_model in sub_models:
        #                 # take the splits and apply them to the corresponding datasets
        #                 if ss_type == 'acceptor':
        #                     X_trn, y_trn = encoded_data[sub_model][0][train], encoded_data[sub_model][1][train]
        #                     X_val, y_val = encoded_data[sub_model][0][test], encoded_data[sub_model][1][test]
        #                 if ss_type == 'donor':
        #                     X_trn, y_trn = encoded_data[sub_model][2][train], encoded_data[sub_model][3][train]
        #                     X_val, y_val = encoded_data[sub_model][2][test], encoded_data[sub_model][3][test]
        #
        #                 history, model = build_models.build_single_model(
        #                     network_rows[ss_type][dataset], # splice site rows
        #                     dataset, # dataset name
        #                     sub_model, # sub_model name
        #                     ss_type,# splice site type
        #                     summary, # whether to include the summar
        #                     X_trn, # the training data
        #                     y_trn, # the training labels
        #                     X_val, # the validation data
        #                     y_val, # the validation labels
        #                     fold_num, # the current fold number
        #                     num_folds, # the total fold count
        #                     save, # whether to save the model
        #                     )
        #                 # add models and values to the list for use in esplice
        #                 histories_models_data.append(
        #                     (
        #                         history,
        #                         model,
        #                         X_trn,
        #                         y_trn,
        #                         X_val,
        #                         y_val,
        #                     )
        #                 )
        #                 # get model results, values for all epochs
        #                 results[ss_type][dataset][sub_model]['val_acc'].append(history.history['val_accuracy'])
        #                 results[ss_type][dataset][sub_model]['val_loss'].append(history.history['val_loss'])
        #                 results[ss_type][dataset][sub_model]['trn_acc'].append(history.history['accuracy'])
        #                 results[ss_type][dataset][sub_model]['trn_acc'].append(history.history['loss'])
        #
        #
        #             # aggregated predictions into a new dataset and train the classifiers
        #             clf1_s, clf2_s, clf3_s = EnsembleSplice.build(
        #                 histories_models_data,
        #                 batch_size
        #             )
        #             # get model results, values over all folds
        #             results[ss_type][dataset]['esplice']['percep'].append(clf1_s)
        #             results[ss_type][dataset]['esplice']['logre'].append(clf2_s)
        #             results[ss_type][dataset]['esplice']['linsvc'].append(clf3_s)
        #
        #             # increment fold_num
        #             fold_num += 1
        #
        #     # calculate means, stds, and averages
        #     # for each splice site type
        #     for ss_type, ss_vals in results.items():
        #         # for each dataset
        #         for dataset, d_vals in ss_vals.items():
        #             # get avg. accuracies across folds
        #             results[ss_type][dataset]['esplice']['per_val_acc_avg'] = avg_out_val_esplice(results[ss_type][dataset]['esplice']['percep'])
        #             results[ss_type][dataset]['esplice']['log_val_acc_avg'] = avg_out_val_esplice(results[ss_type][dataset]['esplice']['logre'])
        #             results[ss_type][dataset]['esplice']['svc_val_acc_avg'] = avg_out_val_esplice(results[ss_type][dataset]['esplice']['linsvc'])
        #             # for each other sub_model
        #             for sub_model, s_vals in d_vals.items():
        #                 for metric, metric_seqs in s_vals.items():
        #                     if sub_model != 'esplice':
        #                         # get means for each epoch
        #                         avg_last_epoch = np.mean(np.asarray([elt[-1] for elt in results[ss_type][dataset][sub_model][metric]]))
        #                         results[ss_type][dataset][sub_model][metric] = avg_last_epoch
        #
        # pass

        # # if save:
        # #     tf.keras.utils.plot_model(model, f'{model_type.upper()}.png', show_shapes=True)
        #
        # # for each fold, each model is trained and the best values are stored
        # # these values are then averaged across folds
        # mean_by_fold =  lambda seq: np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(seq).T)
        # std_by_fold = lambda seq: np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(seq).T)
        #
        # # function for averaging out val_acc, trn_loss etc... over k folds
        # mean_val_esplice = lambda val_list: np.mean(np.asarray(val_list))
        # avg_out_val_other = lambda val_list: np.mean()

    if kwargs['train']:
        pass

    if kwargs['test']:
        pass

    if kwargs['report']:
        pass




    # models=sub_models,
    # esplicers=esplice_archs,
    # datasets=datasets,
    # splice_sites=splice_sites,
    # rows=network_rows,
    # state=args.random_state,
    # train=args.train,
    # validate=args.val,
    # folds=args.k,
    # test=args.test,
    # combine=args.combine,
    # balance=args.bal,
    # imbalance=args.imbal,
    # report=args.report,
    # t_tests=args.t_tests,
    # store_config=args.store_config,







    # if kwargs['train']:

    # the values that are used for graphing
    # means_stds = {
    #     'val_acc':[],
    #     'val_loss':[],
    #     'trn_acc':[],
    #     'trn_loss':[],
    # }
    # # 'val_acc_avg':'',
    # # 'val_loss_avg':'',
    # # 'trn_acc_avg':'',
    # # 'val_loss_avg':''
    #
    # # sub_model dictionary
    # s_models = {
    #     'cnn1':copy.deepcopy(means_stds),
    #     'dnn1':copy.deepcopy(means_stds),
    #     'esplice':{'percep':[], 'logre':[], 'linsvc':[], 'per_val_acc_avg':'', 'log_val_acc_avg':'', 'svc_val_acc_avg':''}, # just val acc for esplice
    #     'rnn1':copy.deepcopy(means_stds),
    #     'rnn2':copy.deepcopy(means_stds),
    #     'rnn3':copy.deepcopy(means_stds),
    #     'cnn2':copy.deepcopy(means_stds),
    #     'cnn3':copy.deepcopy(means_stds),
    #     'cnn4':copy.deepcopy(means_stds),
    #     'cnn5':copy.deepcopy(means_stds),
    #     'dnn2':copy.deepcopy(means_stds),
    #     'dnn3':copy.deepcopy(means_stds),
    # }
    #
    # # initialize dictionary of results
    # to_run = dict([(dataset,s_models) for dataset in datasets])
    #
    # # results dictionary
    # results = dict([(ss_type, copy.deepcopy(to_run)) for ss_type in splice_sites])
    #
    # # get a metrics per sub model method
    # evals = dict(
    #     [
    #         (sub_model, {
    #             'f1':'', 'precision':'',
    #             'sensitivity':'', 'specificity':'',
    #             'recall':'', 'mcc':'',
    #             'err_rate':''
    #         }) for sub_model in sub_models
    #     ]
    # )
    #
    # if val: # if performing vaidation
    #     # function for extracting mean_over_epoch and std_over_epoch
    #     mean_by_epoch =  lambda seq: np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(seq).T)
    #     std_by_epoch = lambda seq: np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(seq).T)
    #     # function for averaging out val_acc, trn_loss etc... over k folds
    #     avg_out_val_esplice = lambda val_list: np.mean(np.asarray(val_list))
    #     avg_out_val_other = lambda val_list: np.mean()
    #
    #     # for each dataset
    #     for dataset in datasets:
    #         encoded_data = dict([(s_model, '') for s_model in sub_models]) # encode data for each sub_model
    #         # for each sub_model
    #         for sub_model in sub_models:
    #             # encode the data
    #             encoded_data[sub_model] = encode_data.encode(dataset, sub_model, bal, test)
    #         for ss_type in splice_sites:
    #             if ss_type == 'acceptor':
    #                 # index [:2] gets (acc_x, acc_y) of (acc_x, acc_y, don_x, don_y)
    #                 X_train, y_train = list(encoded_data.values())[0][:2]
    #             if ss_type == 'donor':
    #                 # which encoded values chosen here are arbitrary,
    #                 # they're just used for the split
    #                 X_train, y_train = list(encoded_data.values())[0][2:]
    #
    #             # this loop inspired by https://www.machinecurve.com/
    #             #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
    #             k_fold = KFold(n_splits=num_folds, shuffle=False)
    #             fold_num = 1
    #             # for train, test split of k folds
    #             for train, test in k_fold.split(X_train, y_train):
    #                 # for each sub_model
    #                 histories_models_data = []
    #                 for sub_model in sub_models:
    #                     # take the splits and apply them to the corresponding datasets
    #                     if ss_type == 'acceptor':
    #                         X_trn, y_trn = encoded_data[sub_model][0][train], encoded_data[sub_model][1][train]
    #                         X_val, y_val = encoded_data[sub_model][0][test], encoded_data[sub_model][1][test]
    #                     if ss_type == 'donor':
    #                         X_trn, y_trn = encoded_data[sub_model][2][train], encoded_data[sub_model][3][train]
    #                         X_val, y_val = encoded_data[sub_model][2][test], encoded_data[sub_model][3][test]
    #
    #                     history, model = build_models.build_single_model(
    #                         network_rows[ss_type][dataset], # splice site rows
    #                         dataset, # dataset name
    #                         sub_model, # sub_model name
    #                         ss_type,# splice site type
    #                         summary, # whether to include the summar
    #                         X_trn, # the training data
    #                         y_trn, # the training labels
    #                         X_val, # the validation data
    #                         y_val, # the validation labels
    #                         fold_num, # the current fold number
    #                         num_folds, # the total fold count
    #                         save, # whether to save the model
    #                         )
    #                     # add models and values to the list for use in esplice
    #                     histories_models_data.append(
    #                         (
    #                             history,
    #                             model,
    #                             X_trn,
    #                             y_trn,
    #                             X_val,
    #                             y_val,
    #                         )
    #                     )
    #                     # get model results, values for all epochs
    #                     results[ss_type][dataset][sub_model]['val_acc'].append(history.history['val_accuracy'])
    #                     results[ss_type][dataset][sub_model]['val_loss'].append(history.history['val_loss'])
    #                     results[ss_type][dataset][sub_model]['trn_acc'].append(history.history['accuracy'])
    #                     results[ss_type][dataset][sub_model]['trn_acc'].append(history.history['loss'])
    #
    #
    #                 # aggregated predictions into a new dataset and train the classifiers
    #                 clf1_s, clf2_s, clf3_s = EnsembleSplice.build(
    #                     histories_models_data,
    #                     batch_size
    #                 )
    #                 # get model results, values over all folds
    #                 results[ss_type][dataset]['esplice']['percep'].append(clf1_s)
    #                 results[ss_type][dataset]['esplice']['logre'].append(clf2_s)
    #                 results[ss_type][dataset]['esplice']['linsvc'].append(clf3_s)
    #
    #                 # increment fold_num
    #                 fold_num += 1
    #
    #     # calculate means, stds, and averages
    #     # for each splice site type
    #     for ss_type, ss_vals in results.items():
    #         # for each dataset
    #         for dataset, d_vals in ss_vals.items():
    #             # get avg. accuracies across folds
    #             results[ss_type][dataset]['esplice']['per_val_acc_avg'] = avg_out_val_esplice(results[ss_type][dataset]['esplice']['percep'])
    #             results[ss_type][dataset]['esplice']['log_val_acc_avg'] = avg_out_val_esplice(results[ss_type][dataset]['esplice']['logre'])
    #             results[ss_type][dataset]['esplice']['svc_val_acc_avg'] = avg_out_val_esplice(results[ss_type][dataset]['esplice']['linsvc'])
    #             # for each other sub_model
    #             for sub_model, s_vals in d_vals.items():
    #                 for metric, metric_seqs in s_vals.items():
    #                     if sub_model != 'esplice':
    #                         # get means for each epoch
    #                         avg_last_epoch = np.mean(np.asarray([elt[-1] for elt in results[ss_type][dataset][sub_model][metric]]))
    #                         results[ss_type][dataset][sub_model][metric] = avg_last_epoch
    #
    #
    # if not val:
    #     pass


    # plot results
    # if vis:
        # GoOD
        # # x rows equal to splice sites times, x plots equal to datasets,
        # fig, axs = plt.subplots(len(splice_sites), len(datasets), figsize=(10, 10), sharey=True)
        #
        # # set fonts
        # tmfont = {'fontname': "serif"}
        # font = font_manager.FontProperties(family='serif', size=12)
        #
        # if len(splice_sites) == 2:
        #
        #
        #     for ax in axs:
        #         ax.grid(True)
        #         yticks = np.arange(start=0.8, stop=1, step=0.01)
        #         ax.set_yticks(ticks=yticks)
        #         ax.set_ylim(0.8, 1.0)
        # if  len(splice_sites) == 1:
        #     axs.grid(True)
        #     yticks = np.arange(start=0.8, stop=1, step=0.01)
        #     axs.set_yticks(ticks=yticks)
        #     axs.set_ylim(0.8, 1.0)



        # else:
        #     for ax in axs:
        #         ax.grid(True)
        #         yticks = np.arange(start=0.8, stop=1, step=0.01)
        #         ax.set_yticks(ticks=yticks)
        #         ax.set_ylim(0.8, 1.0)

        # axs[0, 0].set_xlabel('NN269')
        # axs[0, 1].set_xlabel('HS3D')
        # axs[1, 0].set_xlabel('NN269')
        # axs[1, 1].set_xlabel('HS3D')

        # axs[0, 0].set_title(f'{sub_models} Acc/Loss for Acc. Balanced NN269, {num_folds}-Folds, {epochs}-epochs'
        # axs[0, 1].set_title(f'{sub_models} Acc/Loss for Acc. Balanced HS3D, {num_folds}-Folds, {epochs}-epochs'
        # axs[1, 0].set_title(f'{sub_models} Acc/Loss for Don. Balanced NN269, {num_folds}-Folds, {epochs}-epochs'
        # axs[1, 0].set_title(f'{sub_models} Acc/Loss for Don. Balanced NN269, {num_folds}-Folds, {epochs}-epochs'





        # plt.grid(True)
        # yticks = np.arange(start=80.0, stop=100.0, step=0.5)
        # plt.yticks(ticks=yticks, fontsize=12, **tmfont)
        # plt.ylim(80.0, 100.0)

        # calculate y-ticks by max values, x-ticks is epochs
        # yticks = np.arange(start=0.0, stop=1.0, step=2/20)


        #plt.xticks(list(range(0, num_folds+1)), fontsize=12, **tmfont)
        #plt.xlim(0, num_folds+2)

        # x,y axis labels and enable grid
        # arch = arch.upper()
        # dataset = dataset.upper()
        # splice_site = splice_site[0].upper()+splice_site[1:]
        # plt.xlabel('Folds', fontsize=20, **tmfont)
        # plt.ylabel('Accuracy/Loss', fontsize=20, **tmfont)
        #
        # # title depends on balanced or imbalanced dataset state
        # if balanced:
        #     plt.title(f'{arch} Acc/Loss for Balanced {dataset} {splice_site} Data, {num_folds}-Folds',  fontsize=18,**tmfont)
        #     plt.legend(loc='best', fontsize=12)
        #     plt.savefig(f'./Graphs/{arch}_bal_{dataset}_{splice_site}_{num_folds}-folds.png')
        # else:
        #     plt.title(f'{arch} Acc/Loss for Imbalanced {dataset} {splice_site} Data, {num_folds}-Folds', fontsize=18,**tmfont)
        #     plt.legend(loc='best', fontsize=12)
        #     plt.savefig(f'./Graphs/{arch}_imbal_{dataset}_{splice_site}_{num_folds}-folds.png')


        ## GOOD
        # sites = {'acceptor':0, 'donor':1}
        # index_set = dict([(ds, index) for index, ds in enumerate(datasets)])
        #
        # # for each splice site type
        # for ss_type, ss_vals in results.items():
        #     # for each dataset
        #     for dataset, d_vals in ss_vals.items():
        #         axs[sites[ss_type]].bar('el_per', results[ss_type][dataset]['esplice']['per_val_acc_avg'])
        #         axs[sites[ss_type]].bar('el_log', results[ss_type][dataset]['esplice']['log_val_acc_avg'])
        #         axs[sites[ss_type]].bar('el_svc', results[ss_type][dataset]['esplice']['svc_val_acc_avg'])
        #         # for each submodel
        #         for sub_model, s_vals in d_vals.items():
        #             if sub_model != 'esplice':
        #                 axs[sites[ss_type]].bar(sub_model, results[ss_type][dataset][sub_model]['val_acc'])
        #
        #
        # plt.savefig(f'./Graphs/f.png')




        # if bal:
        #     axs[0].set_title(f'{sub_models} Acc/Loss for Balanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
        #     axs[1].set_title(f'{sub_models} Acc/Loss for Balanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
        #     fig.legend(loc='best', fontsize=12)
        #     #fig.savefig(f'./Graphs/{arch}_bal_{dataset}_{splice_site}_{num_folds}-folds.png')
        # else:
        #     axs[0].set_title(f'{sub_models} Acc/Loss for Imbalanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
        #     axs[1].set_title(f'{sub_models} Acc/Loss for Imbalanced {dataset} Acceptor Data, {num_folds}-Folds, ',  fontsize=18,**tmfont)
        #     fig.legend(loc='best', fontsize=12)
        #     #plt.savefig(f'./Graphs/{arch}_imbal_{dataset}_{splice_site}_{num_folds}-folds.png')


        # plt.show()

        # # plt.plot(list(range(1, num_folds+1)), values[0], label='val acc')
        # # plt.plot(list(range(1, num_folds+1)), values[1], label='trn acc')
        # # names = 'cnn_mva,cnn_sva,cnn_mvl,cnn_svl,cnn_mta,cnn_sta,cnn_mtl,cnn_stl,dnn_mva,dnn_sva,dnn_mvl,dnn_svl,dnn_mtl,dnn_stl,dnn_mta,dnn_sta,rnn_mva,rnn_sva,rnn_mvl,rnn_svl,rnn_mta,rnn_sta,rnn_mtl,rnn_stl'
        # # names = names.split(',')
        # # values = values[2:]
        # # # colors = ['blue','blue','red','red','green','green','orange','orange','brown','brown','black','black','yellow','yellow','cyan','cyan']
        # # for i in range(len(values), 2):
        # #     # print(values[i], values[i+1])
        # #     errorbar(fig, list(range(1, num_folds+1)), values[i], values[i+1], names[i])
        # #names = 'e_val_acc,e_trn_accs,rnn_va,rnn_vl,rnn_ta,rnn_tl,cnn_vl,cnn_va,cnn_tl,cnn_ta,dnn_vl,dnn_va,dnn_tl,dnn_ta'
        # names = 'e_val_acc,e_trn_accs,rnn_va,rnn_ta,cnn_va,cnn_ta,dnn_va,dnn_ta'
        # names = names.split(',')
        #
        # # for index, val in enumerate(values):
        # #     plt.bar(list(range(1, num_folds+1)), [elt*100 for elt in val], label=names[index])
        # # move = np.arange(start=-(num_folds-2)//100, stop=(num_folds-2)//100, step=0.2)
        # # for index, val in enumerate(values):
        # #     plt.bar([elt+move[index] for elt in list(range(1, num_folds+1))], [elt*100 for elt in val], width=0.2, align='center', label=names[index])
        #
        # # x labels values
        #
        # plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*4), [elt*100 for elt in values[0]], width=0.3/4, align='center', label=names[0])
        # plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*3), [elt*100 for elt in values[1]], width=0.3/4, align='center', label=names[1])
        # plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*2), [elt*100 for elt in values[2]], width=0.3/4, align='center', label=names[2])
        # plt.bar(np.asarray(list(range(1, num_folds+1)))-((0.3/4)*1), [elt*100 for elt in values[3]], width=0.3/4, align='center', label=names[3])
        # plt.bar(np.asarray(list(range(1, num_folds+1))), [elt*100 for elt in values[4]], width=0.3/4, align='center', label=names[4])
        # plt.bar(np.asarray(list(range(1, num_folds+1)))+((0.3/4)*1), [elt*100 for elt in values[5]], width=0.3/4, align='center', label=names[5])
        # # plt.bar(np.asarray(list(range(1, num_folds+1)))+((0.3/4)*2), [elt*100 for elt in values[6]], width=0.3/4, align='center', label=names[6])
        # # plt.bar(np.asarray(list(range(1, num_folds+1)))+((0.3/4)*3), [elt*100 for elt in values[7]], width=0.3/4, align='center', label=names[7])
        #
        #
        # # # annotate the sides of the graph with the final epoch values
        # # for var in values:
        # #     plt.annotate(
        # #         '%0.2f' % var[-1],
        # #         xy=(1, var[-1]),
        # #         xytext=(7, 0),
        # #         xycoords=('axes fraction', 'data'),
        # #         textcoords='offset points',
        # #         **tmfont,
        # #     )
        #
        # # calculate y-ticks by max values, x-ticks is epochs
        # # yticks = np.arange(start=0.0, stop=1.0, step=2/20)
        # yticks = np.arange(start=80.0, stop=100.0, step=0.5)
        #
        # plt.yticks(ticks=yticks, fontsize=12, **tmfont)
        # plt.ylim(80.0, 100.0)
        #
        # #plt.xticks(list(range(0, num_folds+1)), fontsize=12, **tmfont)
        # #plt.xlim(0, num_folds+2)
        #
        # # x,y axis labels and enable grid
        # arch = arch.upper()
        # dataset = dataset.upper()
        # splice_site = splice_site[0].upper()+splice_site[1:]
        # plt.xlabel('Folds', fontsize=20, **tmfont)
        # plt.ylabel('Accuracy/Loss', fontsize=20, **tmfont)
        # plt.grid(True)
        # # title depends on balanced or imbalanced dataset state
        # if balanced:
        #     plt.title(f'{arch} Acc/Loss for Balanced {dataset} {splice_site} Data, {num_folds}-Folds',  fontsize=18,**tmfont)
        #     plt.legend(loc='best', fontsize=12)
        #     plt.savefig(f'./Graphs/{arch}_bal_{dataset}_{splice_site}_{num_folds}-folds.png')
        # else:
        #     plt.title(f'{arch} Acc/Loss for Imbalanced {dataset} {splice_site} Data, {num_folds}-Folds', fontsize=18,**tmfont)
        #     plt.legend(loc='best', fontsize=12)
        #     plt.savefig(f'./Graphs/{arch}_imbal_{dataset}_{splice_site}_{num_folds}-folds.png')
        #








                #trn_accuracies, val_accuracies, trn_sub_model_losses, val_sub_model_losses =









            #
            #     # accumulate val, trn loss, acc
            #     site_results[site_type]['val_acc'].append(history.history['val_accuracy'])
            #     site_results[site_type]['val_loss'].append(history.history['val_loss'])
            #     site_results[site_type]['trn_acc'].append(history.history['accuracy'])
            #     site_results[site_type]['trn_loss'].append(history.history['loss'])
            #     # accumulate dataset per fold and models
            #
            #
            #     # iterate fold
            #     fold_num += 1
            #
            # # print new lines
            # print()
            # print('#'*135)
            # print()
            #
            # # means over epochs
            # for key, value in site_results[site_type].items():
            #     site_results[site_type][key] = (mean_by_epoch(site_results[site_type][key]), std_by_epoch(site_results[site_type][key]))
            #
            #
            # encoded_data[sub_model]
            # results = []
            # results.append(
            #     utils.build_single_model(
            #         network_rows[site_type][dataset], # splice site rows
            #         dataset, # dataset name
            #         model_type, # sub_model name
            #         ss_type,# splice site type
            #         summary, # whether to include the summar
            #         X_train, # the training data
            #         y_train, # the training labels
            #         batch_size,# the batch_size for the model
            #         epochs, # the epochs for the model
            #         X_val, # the validation data
            #         y_val, # the validation labels
            #         fold, # the current fold number
            #         num_folds, # the total fold count
            #         save, # whether to save the model
            #         )
            #     )






    # # # for each sub model
    # # for sub_model in sub_models:
    # #     # for each dataset
    # #     for dataset in datasets:
    # #         # pass if the dataset is not used for that sub_model
    # #         if to_run[sub_model][dataset] == '':
    # #             pass
    # #         # otherwise,
    # #         else:
    # #             results[sub_model][dataset] = utils.cross_validation(
    # #                 num_folds,
    # #                 sub_model,
    # #                 splice_sites,
    # #                 dataset,
    # #                 to_run[sub_model][dataset],# encoded data for dataset (ds)
    # #                 network_rows, # donor, acceptor rows for ds
    # #                 evals,
    # #                 summary,
    # #                 config,
    # #                 batch_size,
    # #                 epochs,
    # #                 save,
    # #             )
    # #             # if vis:
    # #
    # # print(results)
    # # return results
    #
    #
    # # pairs (acc_x, acc_y), (don_x, don_y) by site type
    # train_val_pairs = dict([('acceptor',(data[0], data[1])), ('donor',(data[2], data[3]))])
    #
    # # results
    # means_stds = {
    #     'val_acc':[],
    #     'val_loss':[],
    #     'trn_acc':[],
    #     'trn_loss':[],
    # }
    # esplice_values = {
    #     'models':[],
    #     'train_X':[],
    #     'train_y':[],
    #     'val_X':[],
    #     'val_y':[],
    # }
    # site_results[site_type][1]['models'].append(model)
    # site_results[site_type][1]['train_X'].append(X_trn)
    # site_results[site_type][1]['train_y'].append(y_trn)
    # site_results[site_type][1]['val_X'].append(X_val)
    # site_results[site_type][1]['val_y'].append(y_val)
    #
    # site_results = {
    #     'acceptor': copy.deepcopy(means_stds,
    #     'donor': copy.deepcopy(means_stds)
    # }
    #
    # # function for extracting mean_over_epoch
    # mean_by_epoch =  lambda seq: np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(seq).T)
    # std_by_epoch = lambda seq: np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(seq).T)
    #
    # # model builds
    # builds = {'cnn':build_cnn, 'dnn':build_dnn, 'rnn':build_rnn}
    # build_model = builds[model_type]
    #
    # # run cross validation
    # for site_type, site_data in train_val_pairs.items():
    #
    #     X_train, y_train = train_val_pairs[site_type]
    #
    #     # this loop inspired by https://www.machinecurve.com/
    #     #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
    #
    #     k_fold = KFold(n_splits=num_folds, shuffle=False)
    #     fold_num = 1
    #     for train, test in k_fold.split(X_train, y_train):
    #
    #         # partition data accordingly
    #         X_trn, y_trn = X_train[train], y_train[train]
    #         X_val, y_val = X_train[test], y_train[test]
    #
    #         # HAVE A WAY OF CONTROLLING WHICH MODELS GET BUILT
    #         # build model and get history
    #         history, model = cnn_model(
    #             rows[site_type][dataset],
    #             dataset,
    #             model_type,
    #             site_type,
    #             summary,
    #             X_trn,
    #             y_trn,
    #             batch_size,
    #             epochs,
    #             X_val,
    #             y_val,
    #             fold_num,
    #             num_folds,
    #             save
    #         )
    #
    #         history, model = dnn_model(
    #             rows[site_type][dataset],
    #             dataset,
    #             model_type,
    #             site_type,
    #             summary,
    #             X_trn,
    #             y_trn,
    #             batch_size,
    #             epochs,
    #             X_val,
    #             y_val,
    #             fold_num,
    #             num_folds,
    #             save
    #         )
    #
    #         history, model = rnn_model(
    #             rows[site_type][dataset],
    #             dataset,
    #             model_type,
    #             site_type,
    #             summary,
    #             X_trn,
    #             y_trn,
    #             batch_size,
    #             epochs,
    #             X_val,
    #             y_val,
    #             fold_num,
    #             num_folds,
    #             save
    #         )
    #
    #
    #         # accumulate val, trn loss, acc
    #         site_results[site_type]['val_acc'].append(history.history['val_accuracy'])
    #         site_results[site_type]['val_loss'].append(history.history['val_loss'])
    #         site_results[site_type]['trn_acc'].append(history.history['accuracy'])
    #         site_results[site_type]['trn_loss'].append(history.history['loss'])
    #         # accumulate dataset per fold and models
    #
    #
    #         # iterate fold
    #         fold_num += 1
    #
    #     # print new lines
    #     print()
    #     print('#'*135)
    #     print()
    #
    #     # means over epochs
    #     for key, value in site_results[site_type].items():
    #         site_results[site_type][key] = (mean_by_epoch(site_results[site_type][key]), std_by_epoch(site_results[site_type][key]))
    #
    # return (site_results)




    # results = sub_models.run(
    #     datasets,
    #     splice_sites,
    #     s_models,
    #     save,
    #     vis,
    #     iter,
    #     metrics,
    #     summary,
    #     config,
    #     num_folds,
    #     bal,
    #     imbal,
    #     imbal_t,
    #     imbal_f,
    #     batch_size,
    #     epochs
    # )
