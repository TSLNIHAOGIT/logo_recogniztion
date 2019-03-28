import tensorflow as tf
from tensorflow_tools.data_explore import *

def get_data(path=None):
    # Load training and testing datasets.
    images, labels = load_data(path)
    print('类别数：',set(labels))
    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
    # display_images_and_labels(images, labels)
    # display_label_images(images,labels, 26)
    # show_images_size(images)
    data = shringle_images(images, labels)
    return data

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.

with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer.
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels= labels_ph))
    ## Training summary for the current batch_loss

    # Create training op.
    step = tf.Variable(0, trainable=False)
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,global_step=step)

    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.
    init = tf.initialize_all_variables()
    loss_summary = tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()


print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)


# Create a session to run the graph we created.
with tf.Session(
        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True),
        graph=graph) as session:
    ROOT_PATH = 'F:/陶士来文件/tsl_python_project/model_datas'
    train_data_dir = os.path.join(ROOT_PATH, "logo_recogniztion/BelgiumTSC_Training/Training")
    test_data_dir = os.path.join(ROOT_PATH, "logo_recogniztion/BelgiumTSC_Testing/Testing")

    data=get_data(train_data_dir)
    # First step is always to initialize all variables.
    # We don't care about the return value, though. It's None.
    _ = session.run([init])

    summary_writer = tf.summary.FileWriter('model_train', graph=session.graph)

    #一共有多少个e_poch
    n_epoch = 100
    for i in range(n_epoch):
        _, loss_value,summary,current_step= session.run([train, loss,summary_op,step],
                                    feed_dict={images_ph: data['images_a'], labels_ph: data['labels_a']})
        summary_writer.add_summary(summary, current_step)
        if i % 10 == 0:
            print("Loss: ", loss_value)


    # # Pick 10 random images
    # images32=data['images_a']
    # labels=data['labels_a']
    # sample_indexes = random.sample(range(len(images32)), 10)
    # sample_images = [images32[i] for i in sample_indexes]
    # sample_labels = [labels[i] for i in sample_indexes]
    #
    # # Run the "predicted_labels" op.
    # predicted = session.run([predicted_labels],
    #                         feed_dict={images_ph: sample_images})[0]
    # print(sample_labels)
    # print(predicted)
    #
    # #Display the predictions and the ground truth visually.
    # fig = plt.figure(figsize=(10, 10))
    # for i in range(len(sample_images)):
    #     truth = sample_labels[i]
    #     prediction = predicted[i]
    #     plt.subplot(5, 2,1+i)
    #     plt.axis('off')
    #     color='green' if truth == prediction else 'red'
    #     plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
    #              fontsize=12, color=color)
    #     plt.imshow(sample_images[i])
    #
    # # Load the test dataset.
    # test_datas=get_data(test_data_dir)
    # test_images32=test_datas['images_a']
    # test_labels =test_datas['labels_a']
    #
    # # # Transform the images, just like we did with the training set.
    # # test_images32 = [skimage.transform.resize(image, (32, 32))
    # #                  for image in test_images]

    #有问题
    # display_images_and_labels(test_images32, test_labels)

    # Run predictions against the full test set.
    # predicted = session.run([predicted_labels],
    #                         feed_dict={images_ph: test_images32})[0]
    # # Calculate how many matches we got.
    # match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    # accuracy = match_count / len(test_labels)
    # print("Accuracy: {:.3f}".format(accuracy))