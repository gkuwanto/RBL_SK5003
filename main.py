import argparse
import numpy as np

from lib.file_parser import read_csv


# Menambahkan argumen dalam cli
parser = argparse.ArgumentParser()
parser.add_argument("-obj", "--objective", help="Specify objective function of the optimization. Only MSE and MAE is available", default='MSE')
args = parser.parse_args()

# Memastikan Nilai input argument
if (args.objective) not in ["MSE", "MAE"]:
    raise ValueError(f"Objective Function {args.objective} is not implemented yet")
    
# Membaca Data dari file
data_train = read_csv('data/train.csv')
data_test = read_csv('data/test.csv')

headers = data_train.keys()

# Mengubah data dari string menjadi float
for col in headers:
    if col in ['date', 'id']:
        continue
    data_train[col] = [float(i) for i in data_train[col]]
    if col != 'price':
        data_test[col] = [float(i) for i in data_test[col]]


# TODO: One Hot Encode zipcode

# Membaca Data menjadi format numpy
X_train = np.array([data_train[col] for col in headers 
    if col not in ['date', 'id', 'price', 'zipcode']]).T
y_train = np.array(data_train['price'])

# Preprocessing X_train 
X_mean = np.mean(X_train, axis = 0)
X_std = np.std(X_train, axis = 0)
X_train = (X_train - X_mean)/X_std
X_train = np.concatenate([X_train, np.ones(X_train[:, 0:1].shape)], axis = 1)

# Preprocessing y_train
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = (y_train - y_mean)/y_std


# Inisialisasi Parameter Model
theta = np.random.normal(size = X_train[0, :].shape)

# Penentuan Loss dan Gradient tergantung input
if args.objective == 'MSE':
    loss = lambda x, y, theta: 0.5 * (np.dot(theta, x) - y)**2
    gradient = lambda x, y, theta: (np.dot(theta,x) - y) * x
elif args.objective == 'MAE':
    loss = lambda x, y, theta: (np.abs(np.dot(theta, x)-y))
    gradient = lambda x, y, theta: np.sign(np.dot(theta, x)-y) * x


# Memulai Proses SGD
n_epoch = 20
eta = 0.0005
for epoch in range(n_epoch):
    total_loss = 0
    total_sample = len(y_train)
    random_seq = np.random.permutation(np.arange(total_sample))
    for i in random_seq:
        total_loss += loss(X_train[i], y_train[i], theta)
        grad = gradient(X_train[i], y_train[i], theta)
        theta = theta - (eta*grad)
    print(f"Epoch: {epoch} \t Loss: {total_loss/total_sample}")

# Melakukan Preprocessing pada Dataset Test
X_test = np.array([data_test[col] for col in headers 
    if col not in ['date', 'id', 'price', 'zipcode']]).T
X_test = (X_test - X_mean)/X_std
X_test = np.concatenate([X_test, np.ones(X_test[:, 0:1].shape)], axis = 1)

# Menghitung Prediksi
y_pred = np.dot(theta, X_test.T)
# Melakukan postprocessing pada hasil prediksi
y_pred = y_pred * y_std + y_mean

# Menulis Hasil Prediksi pada suatu file
with open('data/prediction.csv', 'w+') as f:
    f.write('id,price\n')
    for id, price in zip(data_test['id'], y_pred):
        f.write(f'{id},{price}\n')
