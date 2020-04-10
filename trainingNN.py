import tensorflow as tf
import numpy as np
import random
#from sklearn.neural_network import MLPClassifier

def NN (X_train, y_train, X_test, y_test, classOfLOS):
    random.seed(1)
    random.shuffle(X_train)
    random.shuffle(y_train)

    print("training with MLP...")
    print("===Test - MLP")
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    for i in range(0, len(y_train)):
        arr = [0] * len(classOfLOS)
        arr[y_train[i]] = 1
        y_train[i] = arr
    y_train = np.array(y_train)
    for i in range(0, len(y_test)):
        arr = [0] * len(classOfLOS)
        arr[y_test[i]] = 1
        y_test[i] = arr
    y_test = np.array(y_test)

    training_epochs = 300
    display_step = 100
    batch_size = 100
    learning_rate = 0.1
    dropout = 0.1
    n_hidden_1 = 100
    n_hidden_2 = n_hidden_1
    n_hidden_3 = n_hidden_1
    n_input = X_train.shape[1]
    n_classes = len(classOfLOS)

    X = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    weights = {
        'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random.normal([n_hidden_3, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.normal([n_hidden_2])),
        'b3': tf.Variable(tf.random.normal([n_hidden_3])),
        'out': tf.Variable(tf.random.normal([n_classes]))
    }

    def MLP(X):
        layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
        #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        #layer_3 = tf.nn.relu(layer_3)
        #layer_3 = tf.nn.dropout(layer_3, dropout)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    logits = MLP(X)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train_op = optimizer.minimize(loss_op)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate, use_locking=False)
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            total_batch = int(X_train.shape[0] / batch_size)
            x_batches = np.array_split(X_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                sess.run(train_op, feed_dict={X: batch_x, y: batch_y})
            #if epoch % display_step == 0:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 y: batch_y})
            print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        print("Optimization Finished!")
        accuracy = sess.run(accuracy, feed_dict={X: X_test,
                                                 y: y_test})
        print("Accuracy:", accuracy)
