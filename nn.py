import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle




# load and preprocess submission data
submission = pd.read_csv('test.csv')
submission = submission.fillna(0.0)
id = submission['PassengerId']
submission_set = submission.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


submission_sex = submission['Sex'] == 'male'
submission_sex = pd.DataFrame(submission_sex, dtype=np.float32)
submission_set['Sex'] = submission_sex

s_embarked = np.empty(len(submission_set))

for index, row in submission_set.iterrows():
    if row['Embarked'] == 'S':
        s_embarked[index] = -1.0
    elif row['Embarked'] == 'C':
        s_embarked[index] = 0.0
    elif row['Embarked'] == 'Q':
        s_embarked[index] = 1.0

submission_set['Embarked'] = s_embarked

x_submission = np.array(submission_set)

# load and preprocess train data
data = pd.read_csv('train.csv')
data = data.fillna(0.0)
data = shuffle(data)
data = data.reset_index()

X = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'index'], axis=1)

print(X.head())

sex = X['Sex'] == 'male'
sex = pd.DataFrame(sex, dtype=np.float32)
X['Sex'] = sex

embarked = np.empty(len(X))

for index, row in X.iterrows():
    if row['Embarked'] == 'S':
        embarked[index] = -1.0
    elif row['Embarked'] == 'C':
        embarked[index] = 0.0
    elif row['Embarked'] == 'Q':
         embarked[index] = 1.0

X['Embarked'] = embarked

# splitting into train and test set
x_train = X[0:int(0.8*len(X))]
y_train = np.empty([int(0.8*len(X)), 2])
x_test = X[int(0.8*len(X)):len(X)]
y_test = np.empty([int(0.2*len(X)+1), 2])

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


# parameters
for i in range(1000):
    tf.reset_default_graph()
    num_of_epochs = 1000
    batch_size = 100
    learning_rate = 10**(-np.random.uniform(1.0, 10.0))
    reg_strength = 10**(-np.random.uniform(0.0, 10.0))
    width = np.random.randint(2,50)
    k = np.random.uniform(0.2,1.0)
    parameters = {'lr': learning_rate, 'reg': reg_strength, 'w':width, 'k': k}
    generate_predictions = False

    # network architecture
    x = tf.placeholder(tf.float32, [None, 7], name='x-input')
    y_true = tf.placeholder(tf.float32, [None, 2], name='y-input')
    k_prob = tf.placeholder(tf.float32, name='keep_prob')

    W = tf.Variable(tf.truncated_normal([7, width], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[1, width]))

    w1 = tf.Variable(tf.truncated_normal([width, 2], stddev=0.1))

    b1 = tf.Variable(tf.constant(0.1, shape=[1, 2]))

    out1 = tf.nn.relu(tf.add(tf.matmul(x,W), b))
    dropped = tf.nn.dropout(out1, k_prob)

    y = tf.nn.softmax(tf.add(tf.matmul(dropped, w1), b1))

    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_true)
    cost = tf.reduce_mean(diff) + (tf.nn.l2_loss(W) +tf.nn.l2_loss(w1))*reg_strength
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # evaluation

    correct_pred = tf.equal(tf.argmax(y_true, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

    auc, auc_update_op = tf.contrib.metrics.streaming_auc(y, y_true)

    tf.scalar_summary('auc', auc)

    summary = tf.merge_all_summaries()

    # training
    # create session
    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        # main training loop
        for epoch_i in range(num_of_epochs):

            batch_count = len(train_x) // batch_size

            # supplying examples in batches
            start = 0
            for batch_i in range(batch_count):

                end = start + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                start += batch_size
                # run optimizer once on batch
                _ = sess.run(
                    [optimizer],
                    feed_dict={x: batch_x, y_true: batch_y, k_prob: k}
                )

                # run eavaluations
                __, s = sess.run(
                    [auc_update_op, summary],
                    feed_dict={x: batch_x, y_true: batch_y, k_prob:1.0}
                )

            # save evaluations (this could be merged with previous step
            loss = sess.run(cost,feed_dict={x: train_x, y_true: train_y, k_prob:1.0})
            acc_train = sess.run(accuracy, feed_dict={x: train_x, y_true: train_y, k_prob:1.0})
            acc_test = sess.run(accuracy, feed_dict={x: test_x, y_true: test_y, k_prob:1.0})
            auc_train = sess.run(auc, feed_dict={x: train_x, y_true: train_y, k_prob:1.0})
            auc_test = sess.run(auc, feed_dict={x: test_x, y_true: test_y, k_prob:1.0})

            # print current evaluations
            print
            print('Epoch', epoch_i + 1, 'completed out of', num_of_epochs)
            print('Loss', loss)
            print('AUC_train:', auc_train, 'ROC_test', auc_test)
            print('accuracy_train:', acc_train, 'accuracy_test', acc_test)

        # save all to output file

        with open("Output5.txt", "a") as text_file:
                 text_file.write(
                        "epoch: {}, roc_train: {}, roc_test: {}, acc_train: {}, acc_test: {}, parameters: {}\n".format(
                            str(epoch_i + 1),
                            str(auc_train),
                            str(auc_test),
                            str(acc_train),
                            str(acc_test),
                            str(parameters)
                        )
                    )

        print("training complete!")
        if generate_predictions:
            print("generating predictions for submission")
            predictions = y.eval(feed_dict={x: x_submission, k_prob:1.0})
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

            print(submission.head())
            submission.to_csv("submission4.csv", index=False)


