#!/bin/bash

# ===================== 配置项（根据你的需求修改） =====================
# 要监控的GPU编号（0和1，对应两张GPU）
GPU_IDS=(0 1)
# 空闲阈值（MB）：低于该值视为空闲
FREE_THRESHOLD=10
# 要执行的Python脚本路径（绝对路径！）
PYTHON_SCRIPT="/users/rh/DistServe/examples/offline.py"
# Python解释器路径（可通过which python/which python3查看）
PYTHON_BIN="/users/rh/miniconda3/envs/distserve/bin/python"
# 日志文件路径（记录执行日志，方便排查问题）
LOG_FILE="/users/rh/tmp/gpu_monitor.log"
# =====================================================================

# 标记：是否所有GPU都空闲
all_free=true

# 遍历每个GPU，检查内存占用
for gpu_id in "${GPU_IDS[@]}"; do
    # 解析nvidia-smi输出，获取该GPU的已用内存（单位：MB）
    # nvidia-smi参数说明：
    # -i $gpu_id：指定GPU编号
    # --query-gpu=memory.used：仅查询已用内存
    # --format=csv,noheader,nounits：输出格式为纯数字（无表头、无单位）
    used_memory=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # 去除输出中的空格（避免解析错误）
    used_memory=$(echo $used_memory | xargs)
    
    # 检查是否为数字（防止nvidia-smi输出异常）
    if ! [[ "$used_memory" =~ ^[0-9]+$ ]]; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') - 错误：GPU $gpu_id 状态解析失败，输出：$used_memory" >> $LOG_FILE
        all_free=false
        break
    fi
    
    # 判断是否超过阈值
    if [ $used_memory -ge $FREE_THRESHOLD ]; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') - GPU $gpu_id 占用内存 $used_memory MB（≥阈值$FREE_THRESHOLD），不满足条件" >> $LOG_FILE
        all_free=false
        break
    else
        echo "$(date +'%Y-%m-%d %H:%M:%S') - GPU $gpu_id 占用内存 $used_memory MB（<阈值$FREE_THRESHOLD），空闲" >> $LOG_FILE
    fi
done

# 所有GPU都空闲时执行Python代码
if [ "$all_free" = true ]; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') - 所有GPU都空闲，执行Python脚本：$PYTHON_SCRIPT" >> $LOG_FILE
    # 执行Python脚本（如需指定GPU，可加CUDA_VISIBLE_DEVICES）
    CUDA_VISIBLE_DEVICES=0,1 $PYTHON_BIN $PYTHON_SCRIPT >> $LOG_FILE 2>&1
    # 记录执行结果
    if [ $? -eq 0 ]; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') - Python脚本执行成功" >> $LOG_FILE
    else
        echo "$(date +'%Y-%m-%d %H:%M:%S') - Python脚本执行失败" >> $LOG_FILE
    fi
fi