# /data/hxy/ppo_50pct_launcher.py
import sys, runpy
import torch

# 将每张可见卡的 PyTorch 显存上限设置为 50%
for i in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(0.5, i)

# 把当前命令行参数原样传给 verl.trainer.main_ppo
# 等价于：python -m verl.trainer.main_ppo <你的所有 hydra 参数...>
sys.argv = ["-m", "verl.trainer.main_ppo"] + sys.argv[1:]
runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
