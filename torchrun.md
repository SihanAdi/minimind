# 常用参数
```
--nproc-per-node	每个节点（机器）上启动的进程数
--nnodes	参与训练的总节点（机器）数
--node-rank	当前机器的编号
--rdzv-endpoint 主节点地址
--rdzv-backend	分布式协调后端   默认为 c10d，适用于大多数场景，无需额外启动协调服务器
--rdzv-id   任务的唯一标识符    多机训练时必须设置
```

# 经典多机多卡
```
# 在主节点（IP: 192.168.1.100）上执行
torchrun \
    --nproc-per-node=4 \
    --nnodes=2 \
    --node-rank=0 \
    --rdzv-endpoint=192.168.1.100:29500 \
    --rdzv-backend=c10d \
    --rdzv-id=my_training_job \
    train.py

# 在第二台机器上执行
torchrun \
    --nproc-per-node=4 \
    --nnodes=2 \
    --node-rank=1 \  # 注意这里改为1
    --rdzv-endpoint=192.168.1.100:29500 \ # 与主节点一致
    --rdzv-backend=c10d \
    --rdzv-id=my_training_job \ # 与主节点一致
    train.py
```

# 弹性多机训练
```
torchrun \
    --nproc-per-node=4 \
    --nnodes=1:4 \  # 允许1到4个节点动态加入
    --node-rank=0 \
    --rdzv-endpoint=<主节点IP>:29500 \
    --max-restarts=3 \
    --rdzv-id=elastic_job \
    train.py
```