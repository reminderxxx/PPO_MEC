import pandas as pd

path = r"D:\PPO_MEC\data\raw\workflow\alibaba2018\batch_task.csv"

cols = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]

df = pd.read_csv(
    path,
    header=None,          # 关键：文件本身没有表头
    names=cols,           # 手动指定官方字段名
    nrows=10
)

print("columns:", list(df.columns))
print(df.head(10).to_string(index=False))