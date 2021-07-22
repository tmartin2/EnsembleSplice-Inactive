from Models import sub_models

import matplotlib.pyplot as plt # plotting
import matplotlib.font_manager as font_manager # plot fonts
import numpy as np # x,y axes values
from sklearn.model_selection import KFold # cross validation
import copy # splice site results copy
import tensorflow as tf


def acc_build_cnn1(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4, 1)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(
        filters=208,
        kernel_size=(4, 4),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=152,
        kernel_size=(2, 2),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=360,
        kernel_size=(5,5),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(4, 4),
        padding='same'
        )
    )
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=600,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model

    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=43,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7,  mode='min', min_delta=0.001),
            ]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=43,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7,  mode='min', min_delta=0.001),
            ]
        )
    return (history, model)

def acc_build_cnn2(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(
        filters=136,
        kernel_size=5,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv1D(
        filters=344,
        kernel_size=6,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=9, padding="same"))
    model.add(tf.keras.layers.Conv1D(
        filters=232,
        kernel_size=5,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.01,
        decay_steps=300,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model

    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=8,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=8,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def acc_build_cnn3(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):

    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(
        filters=344,
        kernel_size=8,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=9, padding="same"))
    model.add(tf.keras.layers.Conv1D(
        filters=160,
        kernel_size=7,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=460,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def acc_build_cnn4(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(
        filters=144,
        kernel_size=10,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=5, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=460,
        decay_rate=2.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def acc_build_cnn5(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4, 1)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    model.add(tf.keras.layers.Conv2D(
        filters=56,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=48,
        kernel_size=(1, 1),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=400,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(5, 5),
        padding='same'
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=(4,4),
        activation="relu",
        padding="same"
        )
    )

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0025,
        decay_steps=540,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=57,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=57,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def acc_build_dnn1(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=608,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0025),
        )
    )
    model.add(tf.keras.layers.Dense(
        units=576,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0025),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.6))
    model.add(tf.keras.layers.Dense(
        units=512,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.05),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=460,
        decay_rate=1.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'tmp/dnn',
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True
    # )
    # tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/checkpoint',monitor='val_loss', save_best_only=True, mode='min')
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True),
            ]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True),
            ]
        )
    return (history, model)

def acc_build_dnn2(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=672,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(
        units=576,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
        )
    )
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=280,
        decay_rate=1.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model

    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True),
            ]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True),
            ]
        )
    return (history, model)

def acc_build_dnn3(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=704,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.7))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0025,
        decay_steps=600,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=26,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=26,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)


def acc_build_rnn1(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(
        units=288,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.LSTM(
        units=160,
        activation='tanh',
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.6))
    model.add(tf.keras.layers.Dense(
        units=160,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )
    )
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=520,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=33,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=33,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def acc_build_rnn2(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(
        units=320,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.LSTM(
        units=672,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.LSTM(
        units=352,
        activation='tanh',
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=500,
        decay_rate=2.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=37,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=37,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def acc_build_rnn3(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(
        units=576,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.Dense(
        units=640,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.005)
        )
    )
    model.add(tf.keras.layers.LSTM(
        units=224,
        activation='tanh',
        )
    )
    model.add(tf.keras.layers.Dense(
        units=480,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.005)
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0001,
        decay_steps=220,
        decay_rate=1.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=14,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=14,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)


def don_build_cnn1(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4,1)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    model.add(tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=(5,5),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=360,
        kernel_size=(4, 4),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(6, 6),
        padding='same'
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
        )
    )
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.01,
        decay_steps=100,
        decay_rate=1.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def don_build_cnn2(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(
        filters=400,
        kernel_size=5,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv1D(
        filters=216,
        kernel_size=10,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=6, padding="same"))
    model.add(tf.keras.layers.Conv1D(
        filters=344,
        kernel_size=3,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=80,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def don_build_cnn3(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):

    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(
        filters=368,
        kernel_size=5,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding="same"))
    model.add(tf.keras.layers.Conv1D(
        filters=384,
        kernel_size=2,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=6, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0025,
        decay_steps=460,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,# was 1 but I change this
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,# was 1 but I change this
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def don_build_cnn4(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(
        filters=360,
        kernel_size=10,
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0025,
        decay_steps=220,
        decay_rate=1.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def don_build_cnn5(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4,1)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(6, 6),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=288,
        kernel_size=(4, 4),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=264,
        kernel_size=(6, 6),
        activation="relu",
        padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.Conv2D(
        filters=392,
        kernel_size=(2,2),
        activation="relu",
        padding="same"
        )
    )

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=380,
        decay_rate=2.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )

    return (history, model)


def don_build_dnn1(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=352,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0025),
        )
    )
    model.add(tf.keras.layers.Dense(
        units=640,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.05),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.6))
    model.add(tf.keras.layers.Dense(
        units=32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.01,
        decay_steps=300,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model

    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=33,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
        return (history, model)
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=33,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
        return (history, model)



def don_build_dnn2(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=288,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.6))
    model.add(tf.keras.layers.Dense(
        units=384,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0025),
        )
    )
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=540,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=14,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=14,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def don_build_dnn3(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=288,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0025),
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.01,
        decay_steps=400,
        decay_rate=2.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=9,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )

    return (history, model)

def don_build_rnn1(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(
        units=416,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.LSTM(
        units=672,
        activation='tanh',
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(
        units=192,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0025)
        )
    )
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0025,
        decay_steps=580,
        decay_rate=1.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=16,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=16,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )

    return (history, model)

def don_build_rnn2(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(
        units=128,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.LSTM(
        units=224,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.LSTM(
        units=352,
        activation='tanh',
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.005,
        decay_steps=340,
        decay_rate=2.0,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=10,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)]
        )
    return (history, model)

def don_build_rnn3(
    rows,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    model = tf.keras.Sequential()
    input_shape = (rows, 4)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(
        units=608,
        activation='tanh',
        return_sequences=True,
        )
    )
    model.add(tf.keras.layers.Dense(
        units=480,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0025)
        )
    )
    model.add(tf.keras.layers.LSTM(
        units=224,
        activation='tanh',
        )
    )
    model.add(tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.0025,
        decay_steps=500,
        decay_rate=0.5,
        staircase=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
            tf.keras.metrics.BinaryCrossentropy(name='bin_cross'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    if kwargs['test']:
        return model
    if kwargs['validate']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=33,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=0.001, restore_best_weights=True)
            ]
        )
    if kwargs['train']:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=33,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, min_delta=0.001, restore_best_weights=True)
            ]
        )
    return (history, model)

def build_single_model(
    rows,
    ss_type,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    kwargs,
    ):
    # add something like *args or **kwargs here
    if ss_type == 'acceptor':
        if model == 'cnn1':
            return acc_build_cnn1(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn2':
            return acc_build_cnn2(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn3':
            return acc_build_cnn3(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn4':
            return acc_build_cnn4(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn5':
            return acc_build_cnn5(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'dnn1':
            return acc_build_dnn1(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'dnn2':
            return acc_build_dnn2(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'dnn3':
            return acc_build_dnn3(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'rnn1':
            return acc_build_rnn1(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'rnn2':
            return acc_build_rnn2(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'rnn3':
            return acc_build_rnn3(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
    if ss_type == 'donor':
        if model == 'cnn1':
            return don_build_cnn1(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn2':
            return don_build_cnn2(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn3':
            return don_build_cnn3(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn4':
            return don_build_cnn4(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'cnn5':
            return don_build_cnn5(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'dnn1':
            return don_build_dnn1(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'dnn2':
            return don_build_dnn2(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'dnn3':
            return don_build_dnn3(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'rnn1':
            return don_build_rnn1(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'rnn2':
            return don_build_rnn2(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
        if model == 'rnn3':
            return don_build_rnn3(
                rows,
                X_train,
                y_train,
                X_val,
                y_val,
                kwargs,
            )
    # if model == 'rnn':
    #     return build_rnn(
    #         dataset_row_num,
    #         dataset,
    #         model,
    #         splice_type,
    #         summary,
    #         X_train,
    #         y_train,
    #         batch_size,# wont need after tuning
    #         epochs, # wont need after tuning
    #         X_val,
    #         y_val,
    #         fold,
    #         num_folds,
    #         save,
    #     )
