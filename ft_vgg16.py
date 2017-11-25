from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import preprocess_input  # pylint: disable=unused-import
from tensorflow.python.keras._impl.keras.engine.topology import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Flatten
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Dropout
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.utils import layer_utils
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):

  if weights not in {'imagenet', None}:
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization) or `imagenet` '
                     '(pre-training on ImageNet).')

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=48,
      data_format=K.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    img_input = Input(tensor=input_tensor, shape=input_shape)

  # Block 1
  x = Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
  x = Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(x, 0.5)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(x, 0.5)
    x = Dense(classes, activation='softmax', name='predictions')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = Model(inputs, x, name='vgg16')

  # load weights
  if weights == 'imagenet':
    if include_top:
      weights_path = get_file(
          'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models')
    else:
      weights_path = get_file(
          'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models')

    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(5089, activation='softmax', name='predictions'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    """
    if K.backend() == 'theano':
      layer_utils.convert_all_kernels_in_model(model)

    if K.image_data_format() == 'channels_first':
      if include_top:
        maxpool = model.get_layer(name='block5_pool')
        shape = maxpool.output_shape[1:]
        dense = model.get_layer(name='fc1')
        layer_utils.convert_dense_weights_data_format(dense, shape,
                                                      'channels_first')
    """

  return model

"""
if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = VGG16(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    """
