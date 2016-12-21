# =================
# Code for Kaggle competition "Titanic: Machine Learning from Disaster"
# more info: https://www.kaggle.com/c/titanic
# author: Artur Kucia
# =================

import tensorflow as tf
import numpy as np
import pandas as pd
from utils import data_preprocessing, data_split
import argparse

# //////////////////////////////////////////
# parsing command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('-train', help='trains new models', type=int)
parser.add_argument('-predict', help='generates predictions using latest best model', type=str)
args = parser.parse_args()

# //////////////////////////////////////////
# parameters 1/2
generate_predictions = False
train = False
sub_file = False
max_iter = 1

if args.predict:
    generate_predictions = True
    sub_file = args.predict
if args.train:
    train = True
    max_iter = args.train

# //////////////////////////////////////////
# load and pre process submission data
if generate_predictions:

    submission_set = data_preprocessing('test.csv')
    x_submission = np.array(submission_set)

# //////////////////////////////////////////
# load and pre process train and test data
if train:
    X = data_preprocessing('train.csv')
    # splitting into train and test set
    x_train, y_train, x_test, y_test = data_split(X, 0.6)

    # one-hot encoding of labels
    i = 0
    for index, row in x_train.iterrows():
        if row['Survived'] == 1:
            y_train[i] = [1, 0]
        else:
            y_train[i] = [0, 1]
        i += 1

    i = 0
    for index, row in x_test.iterrows():
        if row['Survived'] == 1:
            y_test[i] = [1, 0]
        else:
            y_test[i] = [0, 1]
        i += 1

    # removing labels
    train_x = np.array(x_train.drop(['Survived'], axis=1))
    test_x = np.array(x_test.drop(['Survived'], axis=1))

    train_y = np.array(y_train)
    test_y = np.array(y_test)

# //////////////////////////////////////////
# defining the architecture
for i in range(max_iter):

    if train:
        # //////////////////////////////////////////
        # parameters 2/2
        # using random parameters for training
        tf.reset_default_graph()
        num_of_epochs = 5000
        batch_size = 100
        learning_rate = 10**(-np.random.uniform(2, 5))
        reg_strength = 10**(-np.random.uniform(0.0, 5.0))
        width = np.random.randint(10, 20)
        k = np.random.uniform(0.65, 0.75)
        parameters = {'lr': learning_rate, 'reg': reg_strength, 'w': width, 'k': k}
        print(parameters)

    else:
        # //////////////////////////////////////////
        # load the width of hidden layers from log file
        try:
            with open("models/best_model_log.txt", "r") as log_file:
                print("Reading model parameters from log file...")
                line = log_file.readline()
                w_string = line.split("'w':")[1]
                w_string = w_string.replace("}", "")
                width = int(w_string)
                learning_rate = 0
                reg_strength = 0
        except Exception as e:
            print(e)
            print("No models found!")

    # //////////////////////////////////////////
    # network architecture

    # //////////////////////////////////////////
    # network input, placeholders
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, [None, 7], name='x-input')
        y_true = tf.placeholder(tf.float32, [None, 2], name='y-input')
        k_prob = tf.placeholder(tf.float32, name='keep_prob')

    # //////////////////////////////////////////
    # first hidden layer
    with tf.name_scope("Hidden_layer_1") as scope:
        W = tf.get_variable(shape=[7, width], initializer=tf.contrib.layers.xavier_initializer(),name='W1')
        b = tf.Variable(tf.constant(0.1, shape=[1, width]), name='b')
        out = tf.nn.relu(tf.add(tf.matmul(x, W), b))
    dropped = tf.nn.dropout(out, k_prob)

    # //////////////////////////////////////////
    # second hidden layer
    with tf.name_scope("Hidden_layer_2") as scope:
        w1 = tf.get_variable(shape=[width, width], initializer=tf.contrib.layers.xavier_initializer(),name='W2')
        b1 = tf.Variable(tf.constant(0.1, shape=[1, width]), name='b1')
        out2 = tf.nn.relu(tf.add(tf.matmul(dropped, w1), b1))
    dropped2 = tf.nn.dropout(out2, k_prob)

    # //////////////////////////////////////////
    # output layer
    with tf.name_scope("Output_layer") as scope:
        w2 = tf.get_variable(shape=[width, 2], initializer=tf.contrib.layers.xavier_initializer(), name='W3')
        b2 = tf.Variable(tf.constant(0.1, shape=[1, 2]), name='b')
        y = tf.nn.softmax(tf.add(tf.matmul(dropped2, w2), b2))

    # //////////////////////////////////////////
    # cost function
    with tf.name_scope("Cost_function")as scope:
        diff = tf.nn.softmax_cross_entropy_with_logits(y, y_true)
        cost = tf.reduce_mean(diff) + (tf.nn.l2_loss(W) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))*reg_strength

    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.999).minimize(cost)

    # //////////////////////////////////////////
    # evaluation
    with tf.name_scope("Evaluation") as scope:
        correct_pred = tf.equal(tf.argmax(y_true, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        auc, auc_update_op = tf.contrib.metrics.streaming_auc(y, y_true)

    # //////////////////////////////////////////
    # model saver
    saver = tf.train.Saver(tf.global_variables())

    # //////////////////////////////////////////
    # creating tensorflow session
    with tf.Session() as sess:

        # //////////////////////////////////////////
        # TRAINING
        if train:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # main training loop
            for epoch_i in range(num_of_epochs):
                # supplying examples in batches
                batch_count = len(train_x) // batch_size
                start = 0
                for batch_i in range(batch_count):
                    end = start + batch_size
                    batch_x = train_x[start:end]
                    batch_y = train_y[start:end]
                    start += batch_size
                    # run optimizer once on every batch
                    _ = sess.run(
                        [optimizer],
                        feed_dict={x: batch_x, y_true: batch_y, k_prob: k}
                    )
                    # run evaluations
                    __ = sess.run(
                        [auc_update_op],
                        feed_dict={x: batch_x, y_true: batch_y, k_prob: 1.0}
                    )

                # saving evaluations
                loss = sess.run(cost, feed_dict={x: train_x, y_true: train_y, k_prob: 1.0})
                acc_train = sess.run(accuracy, feed_dict={x: train_x, y_true: train_y, k_prob: 1.0})
                acc_test = sess.run(accuracy, feed_dict={x: test_x, y_true: test_y, k_prob: 1.0})
                auc_train = sess.run(auc, feed_dict={x: train_x, y_true: train_y, k_prob: 1.0})
                auc_test = sess.run(auc, feed_dict={x: test_x, y_true: test_y, k_prob: 1.0})

                # print current evaluations
                if (epoch_i+1) % 100 == 0 or epoch_i == 0:
                    print
                    print('Epoch {} completed out of {}'.format(epoch_i+1, num_of_epochs))
                    print('Loss {}'.format(loss))
                    print('AUC_train: {}, AUC_test: {}'.format(auc_train, auc_test))
                    print('accuracy_train: {}, accuracy_test: {}'.format(acc_train, acc_test))

            print("training complete!")
            # comparing the model with the currently best
            best_acc = 0.0
            try:
                with open("models/best_model_log.txt", "r") as log_file:
                    line = log_file.readline()
                    best_acc_train = line.split(',')[1].split(' ')[2]
                    best_acc_train = float(best_acc_train)

                    best_acc_test = line.split(',')[2].split(' ')[2]
                    best_acc_test = float(best_acc_test)
                    best_acc = np.sqrt(best_acc_test*best_acc_train)
            except Exception as e:
                print(e)
                print("No models found!")

            # if better model was found -> overwrite
            if np.sqrt(acc_test*acc_train) > best_acc:
                print("Better model has been found!")
                with open("models/best_model_log.txt", "r") as log_file:
                    content = log_file.read()
                with open("models/best_model_log.txt", "w") as log_file:
                    log_file.write(
                        "epochs: {}, acc_train: {}, acc_test: {}, parameters: {}\n".format(
                            str(num_of_epochs), str(acc_train), str(acc_test), str(parameters)
                        )
                    )
                    log_file.write(content)
                save_path = saver.save(sess, "models/best_model.data")
                print("Model saved in file: %s" % save_path)
        # //////////////////////////////////////////
        # LOADING MODEL
        if generate_predictions:
            # restoring model from file
            saver.restore(sess, "models/best_model.data")
            print("Model restored.")

            print("Generating predictions for submission.")
            # evaluating network output
            predictions = y.eval(feed_dict={x: x_submission, k_prob: 1.0})
            # converting one-hot encoding to classes
            output = np.empty([len(predictions)])
            i = 0
            for row in predictions:
                if row[0] > row[1]:
                    output[i] = 1
                else:
                    output[i] = 0
                i += 1

            d = {'PassengerId': np.array(id).tolist(), 'Survived': output.tolist()}
            submission = pd.DataFrame(d, dtype=np.int32)
            print(submission.head(10))
            # save submission to file
            if sub_file:
                submission.to_csv(sub_file, index=False)
                print("predictions save to " + sub_file)


