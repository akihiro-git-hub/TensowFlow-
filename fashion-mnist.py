import tensorflow as tf
import matplotlib.pyplot as plt

print('データの取得')
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('ラベルのクラスに対応するものを作成しておく')
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

#subplotの3つめの引数はインデックスの指定。左上を１といて横に1つ動くとインデックスが１つ大きくなる。
#ticksはラベルの間隔の事。noneだとそのままの値を出す。
#cmapのbinaryはグレースケール
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(train_images[i],cmap=plt.cm.binary)
  plt.xlabel(class_name[train_labels[i]])
plt.show()

print('層の構築を行っていく')

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
