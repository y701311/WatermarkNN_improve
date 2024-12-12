import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from helpers.loaders import getwmloader, getdataloader

def main():
    wm_image_path = "./data/trigger_set/"
    wm_label_path = "my_trigger_set_labels-cifar.txt"
    wmloader, wmloader_noshuffle = getwmloader(wm_image_path, 100, wm_label_path)
    trainloader, testloader, n_classes = getdataloader("cifar10", "./data", "./data", 50000)

    inputs_wm = None
    inputs_train = None
    for inputs, outputs in wmloader_noshuffle:
        inputs_wm = inputs
    for inputs, outputs in trainloader:
        inputs_train = inputs

    # print(inputs_train.shape)
    # print(inputs_wm.shape)
    # print(inputs_train.min(), inputs_train.max())
    # print(inputs_wm.min(), inputs_wm.max())
    
    # データの前処理: 画像を2次元配列に変換
    inputs_wm = inputs_wm.view(inputs_wm.size(0), -1).numpy()

    # 2. KMeansアルゴリズムの適用
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(inputs_wm)

    # クラスタリング結果
    y_kmeans = kmeans.labels_  # データポイントごとのクラスター割り当て
    centroids = kmeans.cluster_centers_  # 各クラスターの中心点
    # print(y_kmeans)
    # print("0 num: ", np.sum(np.where(y_kmeans == 0, 1, 0)))
    # print("1 num: ", np.sum(np.where(y_kmeans == 1, 1, 0)))
    # print("2 num: ", np.sum(np.where(y_kmeans == 2, 1, 0)))
    # print("3 num: ", np.sum(np.where(y_kmeans == 3, 1, 0)))
    # print("4 num: ", np.sum(np.where(y_kmeans == 4, 1, 0)))
    
    with open(wm_image_path + "my_trigger_set_index.txt", mode="w") as f:
        print(*y_kmeans, file=f)

    # # 3. クラスタリング結果の可視化
    # plt.scatter(inputs_wm[:, 0], inputs_wm[:, 1], c=y_kmeans, cmap='viridis', alpha=0.5, label="Data Points")
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label="Centroids")
    # plt.title("KMeans Clustering")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
