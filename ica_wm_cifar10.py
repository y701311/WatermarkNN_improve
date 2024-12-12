import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

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
    
    images = torch.cat((inputs_train, inputs_wm))

    # データの前処理: 画像を2次元配列に変換
    images = images.view(images.size(0), -1).numpy()  # (N, 3072)
    inputs_wm = inputs_wm.view(inputs_wm.size(0), -1).numpy()

    # # 主成分分析 (PCA)
    # pca = PCA(n_components=100)  # 主成分の数を指定
    # x_pca = pca.fit_transform(images)

    # # 主成分分析の結果をプロット
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.5)
    # plt.title('PCA Projection (First Two Components)')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.savefig("./z_result/pca_result.png")

    # 独立成分分析 (ICA)
    n_components = 5
    ica = FastICA(n_components=n_components, random_state=0)  # 独立成分の数を指定
    x_ica = ica.fit_transform(inputs_wm)

    # 独立成分分析の結果をプロット
    plt.figure(figsize=(8, 6))
    plt.scatter(x_ica[:, 0], x_ica[:, 1], alpha=0.5)
    plt.title('ICA Projection (First Two Components)')
    plt.xlabel('Independent Component 1')
    plt.ylabel('Independent Component 2')
    plt.savefig("./z_result/ica_result.png")
    
    independ_dist = []
    
    # print(f"{x_ica.shape=}")
    coef_max_index0 = 0
    coef_max_index1 = 0
    coef_max_index2 = 0
    coef_max_index3 = 0
    coef_max_index4 = 0
    for i, coef in enumerate(x_ica):
        independ_dist.append(np.argmax(np.abs(coef)))
        # print(f"{i=}, {coef=}, {np.abs(coef)=}, {np.argmax(np.abs(coef))=}")
        
        coef_max_index = np.argmax(np.abs(coef))
        if coef_max_index == 0:
            coef_max_index0 += 1
        elif coef_max_index == 1:
            coef_max_index1 += 1
        elif coef_max_index == 2:
            coef_max_index2 += 1
        elif coef_max_index == 3:
            coef_max_index3 += 1
        elif coef_max_index == 4:
            coef_max_index4 += 1
    
    # print(f"{coef_max_index0=}")
    # print(f"{coef_max_index1=}")
    # print(f"{coef_max_index2=}")
    # print(f"{coef_max_index3=}")
    # print(f"{coef_max_index4=}")
    
    # 正規化
    for i in range(n_components):
        x_ica[:, i] /= np.max(x_ica[:, i])
    
    # print("正規化後")
    coef_max_index0 = 0
    coef_max_index1 = 0
    coef_max_index2 = 0
    coef_max_index3 = 0
    coef_max_index4 = 0
    for i, coef in enumerate(x_ica):
        # print(f"{i=}, {coef=}, {np.abs(coef)=}, {np.argmax(np.abs(coef))=}")
        
        coef_max_index = np.argmax(np.abs(coef))
        if coef_max_index == 0:
            coef_max_index0 += 1
        elif coef_max_index == 1:
            coef_max_index1 += 1
        elif coef_max_index == 2:
            coef_max_index2 += 1
        elif coef_max_index == 3:
            coef_max_index3 += 1
        elif coef_max_index == 4:
            coef_max_index4 += 1
    
    # print(f"{coef_max_index0=}")
    # print(f"{coef_max_index1=}")
    # print(f"{coef_max_index2=}")
    # print(f"{coef_max_index3=}")
    # print(f"{coef_max_index4=}")
    
    with open(wm_image_path + "my_trigger_set_index.txt", mode="w") as f:
        print(*independ_dist, file=f)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
