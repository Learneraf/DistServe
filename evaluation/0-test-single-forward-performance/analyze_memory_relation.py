import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 读取CSV文件
df = pd.read_csv('./result/memory_info.csv')

# 计算 batch_size * input_len
df['tokens'] = df['batch_size'] * df['input_len']

# 按tp值分组
df_tp1 = df[df['tp'] == 1]
df_tp2 = df[df['tp'] == 2]

# 计算相关系数
corr_tp1 = df_tp1['tokens'].corr(df_tp1['single_gpu_peak_memory_MiB'])
corr_tp2 = df_tp2['tokens'].corr(df_tp2['single_gpu_peak_memory_MiB'])

print(f"TP=1, the correlation between tokens and memory: {corr_tp1:.4f}")
print(f"TP=2, the correlation between tokens and memory: {corr_tp2:.4f}")

# 线性回归
slope_tp1, intercept_tp1, r_value_tp1, p_value_tp1, std_err_tp1 = stats.linregress(df_tp1['tokens'], df_tp1['single_gpu_peak_memory_MiB'])
slope_tp2, intercept_tp2, r_value_tp2, p_value_tp2, std_err_tp2 = stats.linregress(df_tp2['tokens'], df_tp2['single_gpu_peak_memory_MiB'])

print(f"\nTP=1, linear regression: memory = {slope_tp1:.4f} * tokens + {intercept_tp1:.4f}")
print(f"R² value: {r_value_tp1**2:.4f}")
print(f"\nTP=2, linear regression: memory = {slope_tp2:.4f} * tokens + {intercept_tp2:.4f}")
print(f"R² value: {r_value_tp2**2:.4f}")

# 绘制图表
plt.figure(figsize=(12, 6))

# 绘制TP=1的数据和回归线
plt.scatter(df_tp1['tokens'], df_tp1['single_gpu_peak_memory_MiB'], color='blue', label='TP=1')
tokens_range = np.linspace(df_tp1['tokens'].min(), df_tp1['tokens'].max(), 100)
plt.plot(tokens_range, slope_tp1 * tokens_range + intercept_tp1, color='blue', linestyle='--')

# 绘制TP=2的数据和回归线
plt.scatter(df_tp2['tokens'], df_tp2['single_gpu_peak_memory_MiB'], color='red', label='TP=2')
tokens_range = np.linspace(df_tp2['tokens'].min(), df_tp2['tokens'].max(), 100)
plt.plot(tokens_range, slope_tp2 * tokens_range + intercept_tp2, color='red', linestyle='--')

plt.title('Analysis of the relationship between memory and tokens')
plt.xlabel('batch_size * input_len (tokens)')
plt.ylabel('single_gpu_peak_memory_MiB')
plt.legend()
plt.grid(True)

# 保存图表
plt.savefig('./result/memory_relation.png')
print('\nChart saved to memory_relation.png')