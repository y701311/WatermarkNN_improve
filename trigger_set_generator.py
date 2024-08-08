"""
pics内の画像はcifar10のデータじゃない？はずだから，ラベルは適当でもいい．
埋め込みなしのモデルが間違えればそれで
"""
import random
import numpy as np

def get_prob_dist(class_num, trigger_set_size, wmembed_mode='avg_max'):
    # 各ラベルの平均値の最大値がwm
    if wmembed_mode == 'avg_max':
        while True:
            prob = np.zeros((trigger_set_size, class_num), dtype=np.float32)
            # 1つラベルを選んで，それを避けるように正答を選んで，検出可能なように割り振る
            watermark_label = random.randint(0, class_num - 1)
            for i in range(trigger_set_size):
                correct_label = random.randint(0, class_num - 2)
                if correct_label >= watermark_label:
                    correct_label += 1
                
                for j in range(class_num):
                    if j == watermark_label:
                        prob[i][j] = 0.3
                    elif j == correct_label:
                        prob[i][j] = 0.5
                    else:
                        prob[i][j] = 0.2 / (class_num - 2)
            
            # print(f'avg: {np.average(prob, axis=0)}')
            max_index = np.argmax((np.average(prob, axis=0)))
            if max_index != watermark_label:
                # 11.5%くらいの確率で失敗する
                continue
            else:
                break
    
    # 分散が小さいのがwm
    elif wmembed_mode == 'var':
        prob = np.zeros((trigger_set_size, class_num), dtype=np.float32)
        # 1つラベルを選んで，それを避けるように正答を選んで，検出可能なように割り振る
        for i in range(trigger_set_size):
            watermark_label = random.randint(0, class_num - 1)
            prob[i][random.randint(0, class_num - 1)] = 1.0
        prob = prob / np.sum(prob)

    return prob

def main():
    path = './data/trigger_set/' + 'my_trigger_set_labels-cifar.txt'
    class_num = 10 # 分類クラスの数
    trigger_num = 100
    trigger_set_size = 5
    trigger_set_num = trigger_num // trigger_set_size
    assert trigger_num % trigger_set_size == 0
    
    random.seed(0)

    with open(path, 'w') as f:
        for i in range(trigger_set_num):
            prob = get_prob_dist(class_num, trigger_set_size, wmembed_mode='avg_max')
            for p in prob:
                print(*p, file=f)
                # print(*p)

if __name__ == '__main__':
    main()
