import tensorflow as tf

#mnistを取得＆スケーリング
mnist = tf.keras.datasets.mnist

(x_train ,y_train) ,(x_test, y_test) = mnist.load_data()

x_train = x_train/ 255
x_test =  x_test/ 255

#レイア構造の構築
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10,activation='softmax')])

#コンパイル
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5,verbose=1)
model.evaluate(x_test, y_test,verbose=1)