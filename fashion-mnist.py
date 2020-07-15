import tensorflow as tf
import matplotlib.pyplot as plt

print('データの取得')
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('------------------------------')

print('以下データ内容の確認')
print('train_image_size:',train_images.shape)
print('train_imaze_length:',len(train_images))
print('train_labels:',train_labels)
print('train_labels_length:',len(train_labels))
print('test_image_size',test_images.shape)
print('test_image_length:',len(test_images))
print('test_labels:',test_labels)
print('test_labels_length:',len(test_labels))

print('以上より訓練データが6万。テストデータ1万。ラベルは０～９まであることが分かる')

print('-------------------------------')

print('imageの確認')

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print('0 ~ 255までで表されているためグレースケールとなっていることが分かる')

print('------------------------------')

print('グレースケールを正規化する')

train_images = train_images/255.0
test_images = test_images/255.0

print('-----------------------------')

print('データの中に何が入っているのかを表示してみる')

