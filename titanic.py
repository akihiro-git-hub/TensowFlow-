import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

#np.set_printoptionsはndarrayの表記を分かりやすくするもの
#precisionは小数点以下の桁数の設定。suppressはTrueで指数表記にする。
np.set_printoptions(precision=3, suppress=True)


with open(train_file_path, 'r') as f:
    names_row = f.readline()


CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)


