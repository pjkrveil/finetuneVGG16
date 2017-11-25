from __future__ import print_function
import tensorflow as tf
import numpy as np

vgg_dir = 'vgg/'
if not isdir(vgg_dir):
    os.makedirs(vgg_dir)

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(vgg_dir + "vgg16.npy"):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Params') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
            vgg_dir + 'vgg16.npy',
            pbar.hook
        )
else:
    print("VGG16 already exists.")

def process_images_thru_vgg(classes, batch_size=10):
    start_time = datetime.now()

    labels = []
    file_list = []
    batch = []

    codes = None

    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)

        for each in classes:
            print("Starting {} images".format(each))
            class_path = image_dir + each
            files = os.listdir(class_path)
            files.remove('.DS_Store')
            for ii, file in enumerate(files, 1):
                file_list.append(each + "/" + file)
                # Add images to the current batch
                img = utils.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
                labels.append(each)

                # Running the batch through the network to get the codes
                if ii % batch_size == 0 or ii == len(files):
                    images = np.concatenate(batch)

                    feed_dict = {input_: images}
                    ## run vgg.relu6 to extract fixed features
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                    # Here I'm building an array of the codes
                    if codes is None:
                        codes = codes_batch
                    else:
                        codes = np.concatenate((codes, codes_batch))

                    # Reset to start building the next batch
                    batch = []
                    print('{} images processed'.format(ii))

    # write files to file
    import csv
    with open('files_v2', 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(file_list)

    # write codes to file
    with open('codes_v2', 'w') as f:
        codes.tofile(f)

    # write labels to file
    with open('labels_v2', 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)

    end_time = datetime.now()
    ts = end_time - start_time
    time_spent = int(ts.total_seconds())
    print("processing images thru vgg16: {} secs".format(time_spent))


def model_run(epochs, keep_prob, checkpoint_dir):
    start_time = datetime.now()

    training_loss_list = []
    val_acc_list = []

    if not isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    iteration = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for x, y in get_batches(train_x, train_y):
                feed = {inputs_: x,
                        labels_: y,
                        keep_prob_0: keep_prob}
                loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                training_rec = {'epoch': e+1, 'training_loss': loss}
                training_loss_list.append(training_rec)

                print("Epoch: {}/{}".format(e+1, epochs),
                      "Iteration: {}".format(iteration),
                      "Training loss: {:.5f}".format(loss))
                iteration += 1

                if iteration % 50 == 0:
                    feed = {inputs_: val_x,
                            labels_: val_y,
                            keep_prob_0: 1.0}
                    val_acc = sess.run(accuracy, feed_dict=feed)

                    val_rec = {'epoch': e+1, 'val_acc': val_acc}
                    val_acc_list.append(val_rec)

                    print("Epoch: {}/{}".format(e+1, epochs),
                          "Iteration: {}".format(iteration),
                          "Validation Acc: {:.4f}".format(val_acc))
        saver.save(sess, checkpoint_dir + "cand_faces_cleaned.ckpt")

    end_time = datetime.now()
    ts = end_time - start_time
    time_spent = int(ts.total_seconds())
    print("fine-tuning vgg16 with my dataset: {} secs".format(time_spent))

    return training_loss_list, val_acc_list


