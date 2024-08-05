C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_c10_scratch.t7 --save_dir checkpoint/ --save_model ftll.t7 --runname fine.tune.last.layer
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_c10_scratch.t7 --save_dir checkpoint/ --save_model ftal.t7 --runname fine.tune.all.layers --tunealllayers
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_c10_scratch.t7 --save_dir checkpoint/ --save_model rtll.t7 --runname reinit.last.layer --reinitll
C:/Users/hara/anaconda3/envs/dev/python.exe fine-tune.py --lr 0.01 --load_path checkpoint/model_c10_scratch.t7 --save_dir checkpoint/ --save_model rtal.t7 --runname reinit_all.layers --reinitll --tunealllayers
