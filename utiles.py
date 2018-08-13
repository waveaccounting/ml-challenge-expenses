import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def normalization(train_data, validation_data):
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    validation_data = scaler.fit_transform(validation_data)
    return train_data, validation_data


def plot(data, label, centers, name):
    plt.figure(name)
    plt.title("{0} Cluster".format(name))
    plt.scatter(data[:, 0], data[:, 1], c=label, edgecolors='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='r', s=50, marker='^')
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")

    for i in range(len(centers)):
        plt.annotate("centroid", centers[i], centers[i])
    plt.show()
