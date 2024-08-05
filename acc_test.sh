C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/fine.tune.last.layerftll.t7 >> result_ftll_nowm.txt
C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/fine.tune.all.layersftal.t7 >> result_ftal_nowm.txt
C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/reinit.last.layerrtll.t7 >> result_rtll_nowm.txt
C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/reinit_all.layersrtal.t7 >> result_rtal_nowm.txt

C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/fine.tune.last.layerftll.t7 --wm_path ./data/trigger_set --wm_lbl labels-cifar.txt --testwm >> result_ftll_wm.txt
C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/fine.tune.all.layersftal.t7 --wm_path ./data/trigger_set --wm_lbl labels-cifar.txt --testwm >> result_ftal_wm.txt
C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/reinit.last.layerrtll.t7 --wm_path ./data/trigger_set --wm_lbl labels-cifar.txt --testwm >> result_rtll_wm.txt
C:/Users/hara/anaconda3/envs/dev/python.exe predict.py --model_path checkpoint/reinit_all.layersrtal.t7 --wm_path ./data/trigger_set --wm_lbl labels-cifar.txt --testwm >> result_rtal_wm.txt
