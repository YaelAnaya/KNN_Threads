import numpy as np
import pandas as pd
from queue import Queue
import threading
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance


class KNN:
    def __init__(self, num_threads):
        self.time = None
        self.y_train = None
        self.X_train = None
        self.k = None
        self.queue = None
        self.predictions = None
        self.num_threads = num_threads

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k_neighbours):
        self.k = k_neighbours
        # Se inicializa el arreglo de predicciones con ceros utilizando el numero de filas de X
        self.predictions = np.zeros(X.shape[0])
        self.queue = Queue()
        # Se agrega a la cola los indices y los valores de X
        for i in range(X.shape[0]):
            self.queue.put((i, X[i, :]))

        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self.thread_predict)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return self.predictions

    def thread_predict(self):
        while not self.queue.empty():
            # Se obtiene el indice y el valor de X
            i, x = self.queue.get()
            distances = distance.cdist([x], self.X_train, 'euclidean').flatten()
            # Se obtienen los indices de los k vecinos mas cercanos
            k_indices = distances.argsort()[:self.k]
            # Se obtienen las etiquetas de los k vecinos mas cercanos
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Se obtiene la etiqueta mas repetida
            self.predictions[i] = max(set(k_nearest_labels), key=k_nearest_labels.count)


train_df = pd.read_csv('./dataset/trn_set.csv').dropna()
test_df = pd.read_csv('./dataset/tst_set.csv').dropna()

# split data into X and y (features and target)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Tomar tiempos y anotarlos en un tabla comparando el rendimiento

if __name__ == '__main__':
    # Realizar la comparacion de tiempos para cada numero de threads y posteriormente graficarlos
    # k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    k_values = [1]
    thread_values = [3, 5, 7]

    for k in k_values:
        for threads in thread_values:
            knn = KNN(threads)
            knn.fit(X_train, y_train)
            start = time.time()
            results = knn.predict(X_test, k)
            end = time.time()
            print("K: ", k, "Threads: ", threads, "Time: ", end - start)
            print("Accuracy: ", accuracy_score(y_test, results))
            # Graficar los tiempos obtenidos para cada numero de threads
            plt.plot(threads, end - start, 'ro')

    plt.title('KNN')
    plt.xlabel('Threads')
    plt.ylabel('Time')
    plt.show()




