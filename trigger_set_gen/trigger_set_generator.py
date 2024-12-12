import random
import time
import argparse
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
                        prob[i][j] = 1.0#0.3
                    elif j == correct_label:
                        prob[i][j] = 0.0#0.5
                    else:
                        prob[i][j] = 0.0 / (class_num - 2)#0.2
            
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

def get_subsets(trigger_num, trigger_set_size, trigger_set_num, independ_dist):
    independ_dist_indexed = [[] for _ in range(trigger_set_size)]
    for i, independ_class in enumerate(independ_dist):
        independ_dist_indexed[independ_class].append(i)
    # print(f"{independ_dist_indexed=}")
    
    independ_dist_num_order = []
    for i in range(trigger_set_size):
        independ_dist_num_order.append([i, len(independ_dist_indexed[i])])
    independ_dist_num_order.sort(key=lambda x: x[1], reverse=True)
    # print(f"{independ_dist_num_order=}")
    
    subsets = []
    for i in range(trigger_set_num):
        subset = []
        trigger_used_num = 0
        subset_complete_flag = False
        while True:
            independ_dist_num_order.sort(key=lambda x: x[1], reverse=True)
            for j, (independ_class, num) in enumerate(independ_dist_num_order):
                if len(independ_dist_indexed[independ_class]) == 0:
                    continue
                
                id = random.choice(independ_dist_indexed[independ_class])
                # id = independ_dist_indexed[independ_class][0] # debug
                independ_dist_indexed[independ_class].remove(id)
                subset.append(id)
                independ_dist_num_order[j][1] -= 1
                trigger_used_num += 1
                
                if trigger_used_num == trigger_set_size:
                    subset_complete_flag = True
                    break
            if subset_complete_flag:
                break
        
        # print(f"{independ_dist_num_order=}")
        subset.sort()
        subsets.append(subset)
    
    # print(f"{subsets=}")
    
    return subsets

def subset_to_trigger_subset_id(subsets, trigger_num):
    trigger_subset_id = [-1 for _ in range(trigger_num)]
    
    for subset_id, subset in enumerate(subsets):
        for trigger_id in subset:
            trigger_subset_id[trigger_id] = subset_id
    # print(f"{trigger_subset_id=}")
    
    return trigger_subset_id

def get_probs_from_trigger_subset_id(trigger_subset_id, class_num, trigger_set_num):
    probs = []
    subset_class = [random.randint(0, class_num - 1) for _ in range(trigger_set_num)]
    # print(f"{subset_class=}")
    
    for subset_id in trigger_subset_id:
        prob = np.zeros(class_num, dtype=np.float32)
        index = subset_class[subset_id]
        prob[index] = 1.0
        probs.append(prob)
        # print(f"{prob=}")
    
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_method", default="random")
    args = parser.parse_args()
    
    path = './data/trigger_set/' + 'my_trigger_set_labels-cifar.txt'
    class_num = 10 # 分類クラスの数
    trigger_num = 100
    trigger_set_size = 5
    trigger_set_num = trigger_num // trigger_set_size
    assert trigger_num % trigger_set_size == 0
    
    random.seed(time.time())
    
    if args.gen_method == "sequential":
        # subset id sequential
        print("sequential")
        with open(path, 'w') as f:
            for i in range(trigger_set_num):
                prob = get_prob_dist(class_num, trigger_set_size, wmembed_mode='avg_max')
                for p in prob:
                    print(i, *p, file=f)
                    # print(*p)
    
    if args.gen_method == "random":
        # subset id random
        print("random")
        prob_randam = []
        for i in range(trigger_set_num):
            prob = get_prob_dist(class_num, trigger_set_size, wmembed_mode='avg_max')
            for p in prob:
                prob_randam.append((i, p))
        random.shuffle(prob_randam)
        with open(path, 'w') as f:
            for i, p in prob_randam:
                print(i, *p, file=f)
                # print(*p)
    
    if args.gen_method == "distributed":
        # subset id distributed
        print("distributed")
        independ_dist = None
        with open("./data/trigger_set/" + "my_trigger_set_index.txt", mode="r") as f:
            independ_dist = list(map(int, f.readline().split()))
        # print(f"{independ_dist=}")
        
        subsets = get_subsets(trigger_num, trigger_set_size, trigger_set_num, independ_dist)
        trigger_subset_id = subset_to_trigger_subset_id(subsets, trigger_num)
        probs = get_probs_from_trigger_subset_id(trigger_subset_id, class_num, trigger_set_num)
        
        with open(path, 'w') as f:
            for subset_id, p in zip(trigger_subset_id, probs):
                print(subset_id, *p, file=f)
                # print(*p)

if __name__ == '__main__':
    main()
