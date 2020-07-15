import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print('データの取得')
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('ラベルのクラスに対応するものを作成しておく')
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
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

#subplotの3つめの引数はインデックスの指定。左上を１といて横に1つ動くとインデックスが１つ大きくなる。あくまで表示する場所をインデックスで指定しているだけ
#ticksはラベルの間隔の事。noneだとそのままの値を出す。
#cmapのbinaryはグレースケール
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(train_images[i],cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()

print('層の構築を行っていく')

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print('------------------------')
print('訓練の開始')
model.fit(train_images,train_labels,epochs=5,verbose=2)
print('訓練完了')

print('------------------------')
print('損失関数と正解率を表示する')
test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('test_loss:',test_loss)
print('test_accuracy:',test_acc)

print('------------------------')
print('作成した分類器で予測する')
predictions = model.predict(test_images)
print('1枚目の各クラスの確率:',predictions[0])
print('最も確率が高いクラス',np.argmax(predictions[0]))
print('1枚目の教師データラベル',test_labels[0])
if np.argmax(predictions[0]) == test_labels[0]:
  print('正しく推測されている。')
else:
  print('推測を失敗した。')

print('----------------------')

#イメージの表示と確率の表示をする関数を作成する。予測と教師データが合っていれば青でラベルを表示する
def plot_image(i,predictions_array,true_label,img):
  predictions_array,true_label,img = predictions_array[i],true_label[i],img[i]

  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)
  

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else: 
    color = 'red'

# {:2.0f}はformat関数の文字列指定によるもの。:の右側に有効桁数。有効小数点を指定。今回は2桁で小数点以下は切り捨てとなる。
#np.argmax()は引数の中で一番大きい値のインデックスを取得する
#np.max()は引数の値の中で一番大きい値を取得する。
  plt.xlabel('{} {:2.0f}% (true_label:{})'.format(class_names[predicted_label], 100*np.max(predictions_array),class_names[true_label]),color = color)
  


def plot_value_array(i,predictions_array,true_label):
  predictions_array,true_label = predictions_array[i],true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10),predictions_array,color='#777777')
  plt.ylim(0,1)
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

print('画像と確率をプロットする')
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions,test_labels)
plt.show()

print('--------------------------')

print('テスト画像と予測ラベルを一覧で表示する')

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(12,10))
for i in range (num_images):
  plt.subplot(num_rows,2*num_cols,2*i+1)
  plot_image(i,predictions,test_labels,test_images)
  plt.subplot(num_rows,2*num_cols,2*i+2)
  plot_value_array(i,predictions,test_labels)
plt.show()
print('---------------------------------------------')

print('訓練済みモデルを使って一枚の画像のみを予測する')

img = test_images[0]
print(img.shape)

#tf.kerasは引数が3つ必要。理由は、バッチ単位で行うことになっているから。
#imgのサイズは(28,28)だが、一枚だと(1,28,28)でないといけない。

print('次元を整形する')
#np.expand_dimsは新たに次元を追加するメソッド
img = (np.expand_dims(img,0))
print(img.shape)

print('--------------')

print('各クラスの確率を出力')
prediction_single = model.predict(img)
print(prediction_single)

#ratationでラベルを見やすいように45度傾ける。
plot_value_array(0,prediction_single,test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

print('予測ラベルは:',np.argmax(prediction_single))
print('正解ラベルは:',test_labels[0])
if test_labels[0]==np.argmax(prediction_single):
  print('これにてチュートリアル完成！！！！！')

