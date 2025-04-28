# ParaOpt
## ParaOpt.py 

普通代码

串行：python ParaOpt.py --model_name cnn --training_mode serial --optimizer_type sgd --ema_decay 0.85 --threshold 1e-6 --max_step 60000

模拟并行：python ParaOpt.py --model_name cnn --training_mode parallel --optimizer_type sgd --ema_decay 0.85 --threshold 1e-6 --max_step 60000

## ParaOpt_Sweep.py

批量参数代码

## ParaOpt_mul_v1/v2.py

实时并行代码

实际并行：python ParaOpt_mul_v1.py --P 7 

         --P 选择窗口大小 (P+1即GPU数量) 即选择GPU数量

         python ParaOpt_mul_v2.py --P 7
         
         --P 选择窗口大小 (P+1即GPU数量)
