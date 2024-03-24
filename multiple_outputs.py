import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

scaler = MinMaxScaler()

df = pd.read_csv(r"https://datahub.io/machine-learning/autos/r/autos.csv")

num = df.select_dtypes(include=["number"]).columns
cat = df.select_dtypes(include=["object"]).columns

for col in cat:
    df[col] = df[col].astype("category")
    df[col] = df[col].cat.codes


train, test = train_test_split(df, test_size=0.2, random_state=33)

train = train.dropna()
test = test.dropna()

train_x = train.drop(["num-of-cylinders", "price"], axis=1)
test_x = test.drop(["num-of-cylinders", "price"], axis=1)

train_y = (train["price"].to_numpy(), train["num-of-cylinders"].to_numpy())
test_y = (test["price"].to_numpy(), test["num-of-cylinders"].to_numpy())

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

inp = Input(shape=(train_x.shape[1],))
x = Dense(64, activation="relu")(inp)
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
output1 = Dense(1, activation="softmax", name="cylinders")(x)
output2 = Dense(1, name="price")(x)

model = Model(inputs=inp, outputs=[output1, output2])
print(model.summary())


model.compile(
    optimizer="adam",
    loss={"cylinders": "categorical_crossentropy", "price": "mse"},
    metrics={"cylinders": "accuracy", "price": "mse"},
)

history = model.fit(train_x, train_y, epochs=8, batch_size=32, validation_split=0.2)

print(model.evaluate(test_x, test_y))
