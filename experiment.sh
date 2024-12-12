#!/bin/bash

# kmeansとかICAでトリガーの独立性を算出 kmeans_wm.pyとか
# 算出した独立性をもとにサブセットを構成 trigger_set_generator.py
# 学習，攻撃

for i in {1..10}
do

echo "loop ${i}"

# ランダムににサブセットを構成
C:/Users/hara/anaconda3/envs/dev/python.exe x:/R06_5I_YamadaLab/YAMADA/research/WatermarkNN_improve/trigger_set_gen/trigger_set_generator.py --gen_method random
# 学習
C:/Users/hara/anaconda3/envs/dev/python.exe train.py --batch_size 100 --max_epochs 60 --runname train --save_model model_random${i}.t7 --wm_batch_size 2 --wmtrain --wm_lbl my_trigger_set_labels-cifar.txt >> experiment_result/learn_result_random${i}.txt
# 攻撃
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_random${i}.t7 --save_dir checkpoint/ --save_model model_random${i}_ftll.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname fine.tune.last.layer >> experiment_result/learn_result_random${i}_fine.tune.last.layer.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_random${i}.t7 --save_dir checkpoint/ --save_model model_random${i}_ftal.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname fine.tune.all.layers --tunealllayers >> experiment_result/learn_result_random${i}_fine.tune.all.layers.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_random${i}.t7 --save_dir checkpoint/ --save_model model_random${i}_rtll.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname reinit.last.layer --reinitll >> experiment_result/learn_result_random${i}_reinit.last.layer.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_random${i}.t7 --save_dir checkpoint/ --save_model model_random${i}_rtal.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname reinit_all.layers --reinitll --tunealllayers >> experiment_result/learn_result_random${i}_reinit_all.layers.txt
C:/Users/hara/anaconda3/envs/dev/python.exe distillation.py --batch_size 100 --max_epochs 60 --load_path checkpoint/model_random${i}.t7 --runname distillation  --save_model model_random${i}_distillation.t7 --wm_lbl my_trigger_set_labels-cifar.txt >> experiment_result/learn_result_random${i}_distillation.txt


# kmeansでトリガーの独立性を算出
C:/Users/hara/anaconda3/envs/dev/python.exe x:/R06_5I_YamadaLab/YAMADA/research/WatermarkNN_improve/kmeans_wm.py
# 算出した独立性をもとにサブセットを構成
C:/Users/hara/anaconda3/envs/dev/python.exe x:/R06_5I_YamadaLab/YAMADA/research/WatermarkNN_improve/trigger_set_gen/trigger_set_generator.py --gen_method distributed
# 学習
C:/Users/hara/anaconda3/envs/dev/python.exe train.py --batch_size 100 --max_epochs 60 --runname train --save_model model_kmeans${i}.t7 --wm_batch_size 2 --wmtrain --wm_lbl my_trigger_set_labels-cifar.txt >> experiment_result/learn_result_kmeans${i}.txt
# 攻撃
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_kmeans${i}.t7 --save_dir checkpoint/ --save_model model_kmeans${i}_ftll.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname fine.tune.last.layer >> experiment_result/learn_result_kmeans${i}_fine.tune.last.layer.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_kmeans${i}.t7 --save_dir checkpoint/ --save_model model_kmeans${i}_ftal.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname fine.tune.all.layers --tunealllayers >> experiment_result/learn_result_kmeans${i}_fine.tune.all.layers.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_kmeans${i}.t7 --save_dir checkpoint/ --save_model model_kmeans${i}_rtll.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname reinit.last.layer --reinitll >> experiment_result/learn_result_kmeans${i}_reinit.last.layer.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_kmeans${i}.t7 --save_dir checkpoint/ --save_model model_kmeans${i}_rtal.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname reinit_all.layers --reinitll --tunealllayers >> experiment_result/learn_result_kmeans${i}_reinit_all.layers.txt
C:/Users/hara/anaconda3/envs/dev/python.exe distillation.py --batch_size 100 --max_epochs 60 --load_path checkpoint/model_kmeans${i}.t7 --runname distillation  --save_model model_kmeans${i}_distillation.t7 --wm_lbl my_trigger_set_labels-cifar.txt >> experiment_result/learn_result_kmeans${i}_distillation.txt


# ICAでトリガーの独立性を算出
C:/Users/hara/anaconda3/envs/dev/python.exe x:/R06_5I_YamadaLab/YAMADA/research/WatermarkNN_improve/ica_wm_cifar10.py
# 算出した独立性をもとにサブセットを構成
C:/Users/hara/anaconda3/envs/dev/python.exe x:/R06_5I_YamadaLab/YAMADA/research/WatermarkNN_improve/trigger_set_gen/trigger_set_generator.py --gen_method distributed
# 学習
C:/Users/hara/anaconda3/envs/dev/python.exe train.py --batch_size 100 --max_epochs 60 --runname train --save_model model_ica${i}.t7 --wm_batch_size 2 --wmtrain --wm_lbl my_trigger_set_labels-cifar.txt >> experiment_result/learn_result_ica${i}.txt
# 攻撃
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_ica${i}.t7 --save_dir checkpoint/ --save_model model_ica${i}_ftll.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname fine.tune.last.layer >> experiment_result/learn_result_ica${i}_fine.tune.last.layer.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_ica${i}.t7 --save_dir checkpoint/ --save_model model_ica${i}_ftal.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname fine.tune.all.layers --tunealllayers >> experiment_result/learn_result_ica${i}_fine.tune.all.layers.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_ica${i}.t7 --save_dir checkpoint/ --save_model model_ica${i}_rtll.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname reinit.last.layer --reinitll >> experiment_result/learn_result_ica${i}_reinit.last.layer.txt
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_ica${i}.t7 --save_dir checkpoint/ --save_model model_ica${i}_rtal.t7 --wm_lbl my_trigger_set_labels-cifar.txt --runname reinit_all.layers --reinitll --tunealllayers >> experiment_result/learn_result_ica${i}_reinit_all.layers.txt
C:/Users/hara/anaconda3/envs/dev/python.exe distillation.py --batch_size 100 --max_epochs 60 --load_path checkpoint/model_ica${i}.t7 --runname distillation  --save_model model_ica${i}_distillation.t7 --wm_lbl my_trigger_set_labels-cifar.txt >> experiment_result/learn_result_ica${i}_distillation.txt

done
