# ParaOpt
## ParaOpt.py 

普通代码 运行这个文件即可

串行：python ParaOpt.py --model_name cnn --training_mode serial --optimizer_type adamw --ema_decay 0.85 --threshold 1e-6 --max_step 10000

并行：python ParaOpt.py --model_name cnn --training_mode parallel --optimizer_type adamw --ema_decay 0.85 --threshold 1e-6 --max_step 10000

## ParaOpt_Sweep.py

批量参数代码

## ParaOpt_mul.py

实时并行代码
