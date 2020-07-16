import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as tfds

#タイタニック号のデータを取得する

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

#URLを指定してデータセットを取得する。その後任意のファイル名に入れる
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

#np.set_printoptionsはndarrayの表記を分かりやすくするもの
#precisionは小数点以下の桁数の設定。suppressはTrueで指数表記にする。
np.set_printoptions(precision=3, suppress=True)

#openメソッドでtrain.csvを開き、一行ずつ読み込みname_rowへ代入
with open(train_file_path, 'r') as f:
    names_row = f.readline()

#name_rowのメタ文字nを除去し、,ごとに分ける。
CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

#使用しないコラムを指定
drop_columns = ['fare', 'embaek_town']

#for文の中にif文を記載する事を一行で記述
#一行のfor文は　処理 for 変数 in イテラブルとなる。
#if文はcolがdrop_columnsに入っていないときにTrueとなる。
#よって、Trueになったもののみ残る＝使いたいコラム→消したいモノをdrop_columnsへ指定。との流れになる。
columns_to_use = [col for col in CSV_COLUMNS if col not in drop_columns]

#tf.data.experimental.make_csv_datasetはdatasetを作るメソッド
#引数に指定したモノをdatasetの設定にする事が出来る。
dataset = tf.data.experimental.make_csv_dataset(select_columns=columns_to_use)

#今回の求めたいものをone-hotで表現する。今回は生存か死亡かが争点
LABELS = [0,1]
LABEL_COLUMN = 'survived'

#CSV_COLUMNSの中から一つずつ取り出し、survivedと同じか比べ同じではなかったときにTrueになる。→Falseになるのはsurvivrdのみのためsurvivedが消えたSCV_COLUMNSがFEATURE_COLUMNSに代入される。
FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(file_path,batch_size=12,label_name=LABEL_COLUMN,na_value='?',num_epochs=1,ignore_errors=True)
    return dataset

raw_train_data = get_dataset(test_file_path)
raw_test_data = get_dataset(test_file_path)

example, labels = next(iter(raw_train_data))
print("EXAMPLE: \n" , example )
print("LABELS:\n",labels)


