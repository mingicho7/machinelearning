import helper # Celeba 데이터셋을 다운로드하고 압축을 풀기 위한 외부 코드
import problem_unittests as tests # 인공신경망에서 문제가 있는 부분을 찾아내고 테스트하기 위한 외부 코드
%matplotlib inline
import os
from glob import glob
import numpy as np
from matplotlib import pyplot
import warnings
import tensorflow as tf
import cv2

# 모델 인풋을 만드는 function
def model_inputs(image_width, image_height, image_channels, z_dim):
    real_input_images = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], 'real_input_images')
    input_z = tf.placeholder(tf.float32, [None, z_dim], 'input_z')
    learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
    return real_input_images, input_z, learning_rate

# 인풋이 fake인지 real인지 구별하는 신경망 function
def discriminator(images, reuse=False, alpha=0.2, keep_prob=0.5):
    with tf.variable_scope('discriminator', reuse=reuse):

        conv1 = tf.layers.conv2d(images, 64, 5, 2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        lrelu1 = tf.maximum(alpha * conv1, conv1)
        drop1 = tf.layers.dropout(lrelu1, keep_prob)

        conv2 = tf.layers.conv2d(drop1, 128, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2)
        lrelu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = tf.layers.dropout(lrelu2, keep_prob)

        conv3 = tf.layers.conv2d(drop2, 256, 5, 2, 'same', use_bias=False)
        bn3 = tf.layers.batch_normalization(conv3)
        lrelu3 = tf.maximum(alpha * bn3, bn3)
        drop3 = tf.layers.dropout(lrelu3, keep_prob)

        flat = tf.reshape(drop3, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits

# 노이즈로부터 Discriminator에 인풋으로 줄 데이터를 생성하는 신경망 function
def generator(z, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5):
    with tf.variable_scope('generator', reuse=(not is_train)):

        fc = tf.layers.dense(z, 4 * 4 * 1024, use_bias=False)
        fc = tf.reshape(fc, (-1, 4, 4, 1024))
        bn0 = tf.layers.batch_normalization(fc, training=is_train)
        lrelu0 = tf.maximum(alpha * bn0, bn0)
        drop0 = tf.layers.dropout(lrelu0, keep_prob, training=is_train)

        conv1 = tf.layers.conv2d_transpose(drop0, 512, 4, 1, 'valid', use_bias=False)
        bn1 = tf.layers.batch_normalization(conv1, training=is_train)
        lrelu1 = tf.maximum(alpha * bn1, bn1)
        drop1 = tf.layers.dropout(lrelu1, keep_prob, training=is_train)

        conv2 = tf.layers.conv2d_transpose(drop1, 256, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2, training=is_train)
        lrelu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = tf.layers.dropout(lrelu2, keep_prob, training=is_train)

        logits = tf.layers.conv2d_transpose(drop2, out_channel_dim, 5, 2, 'same')

        out = tf.tanh(logits)

        return out

# discriminator와 generator의 loss를 계산하는 function
def model_loss(input_real, input_z, out_channel_dim, alpha=0.2, smooth_factor=0.1):
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))

    input_fake = generator(input_z, out_channel_dim, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(input_fake, reuse=True, alpha=alpha)

    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    return d_loss_real + d_loss_fake, g_loss

# 모델 최적화 function
def model_opt(d_loss, g_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

# generator의 샘플 output을 출력하는 function
def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    return images_grid

# 실제 모델 학습 function
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode, print_every=10, show_every=100):
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3], alpha=0.2)
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, learning_rate, beta1)

    steps = 0
    count = 0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        save_path = saver.save(sess, "/tmp/model.ckpt")
        ckpt = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, save_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch_i in range(epoch_count):
            os.mkdir('output/' + str(epoch_i))
            for batch_images in get_batches(batch_size):
                steps += 1
                batch_images *= 2.0

                # 랜덤으로 샘플에서 노이즈를 추출
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                # 학습 최적화 Process
                sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                sess.run(g_train_opt, feed_dict={input_z: batch_z})

                if steps % print_every == 0:
                    # 각 스텝마다 Generator Loss와 Discriminator Loss를 출력 (학습이 잘 되고 있는지 확인하는 지표)
                    train_loss_d = d_loss.eval({input_real: batch_images, input_z: batch_z})
                    train_loss_g = g_loss.eval({input_z: batch_z})
                    print("Epoch {}/{} Step {}...".format(epoch_i + 1, epoch_count, steps),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % show_every == 0:
                    count = count + 1
                    iterr = count * show_every
                    # 현재 학습 결과 sample output을 png 파일로 저장
                    images_grid = show_generator_output(sess, 25, input_z, data_shape[3], data_image_mode)
                    dst = os.path.join("output", str(epoch_i), str(iterr) + ".png")
                    pyplot.imsave(dst, images_grid)

                # 모델 메타 데이터 저장
                if epoch_i % 10 == 0:
                    if not os.path.exists('./model/'):
                        os.makedirs('./model')
                    saver.save(sess, './model/' + str(epoch_i))

if __name__ == "__main__":
    # 앞서 정의한 function들의 동작 테스트
    tests.test_model_inputs(model_inputs)
    tests.test_discriminator(discriminator, tf)
    tests.test_generator(generator, tf)
    tests.test_model_loss(model_loss)
    tests.test_model_opt(model_opt, tf)

    # 학습 Parameters
    show_n_images = 25
    batch_size = 64
    z_dim = 100
    learning_rate = 0.00025
    beta1 = 0.45
    epochs = 100 # 시간/장비의 제약으로 인해 11번째 epoch 까지만 진행

    # celeba.zip 데이터셋 다운로드
    helper.download_extract('celeba', './celeba')
    mnist_images = helper.get_batch(glob(os.path.join('./celeba', 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
    pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB')) # 샘플 데이터 1개 보여주기

    # 다운로드 받은 celeba 이미지들을 학습시킬 수 있는 데이터셋으로 변환
    celeba_dataset = helper.Dataset('celeba', glob(os.path.join('./celeba', 'img_align_celeba/*.jpg')))

    # GPU 사용 여부 확인
    if not tf.test.gpu_device_name():
        warnings.warn('GPU가 없습니다. - GPU 사용 권장')
    else:
        print('사용중인 GPU: {}'.format(tf.test.gpu_device_name()))

    # 실제 학습시키고 output을 생성하는 부분 (시간이 오래 걸립니다)
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches, celeba_dataset.shape, celeba_dataset.image_mode)

    # 생성된 output png 파일들을 큰 사이즈로 변환시키는 부분
    for f in glob("output/**/*.png"):
        image = cv2.imread(f)
    large = cv2.resize(image, (0, 0), fx=3, fy=3)
    cv2.imwrite(f, large)
