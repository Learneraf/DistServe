## 使用指南

1. **`./1-simulate_dist_run.sh`**  
   用于运行模拟仿真框架，使用 `PROFILE_ENV` 和 `PROFILE_PATH` 两个环境变量来指定 `time_estimator` 使用的配置文件（脚本中已定义），最终得到模拟运行的各项延迟指标，位于 `./result/latency` 路径下。

2. **`./2-merged_analyze_run.sh`**  
   用于将模拟运行得到的结果和真实运行得到的结果进行比对，其中真实运行得到的结果放置在 `exp_data` 路径下，比对得到的结果位于 `./result/slo` 路径下，接下来的所有步骤就是将这些结果绘制为图。

3. **`./3-plot_all_rate_run`**  
   用于绘制 `slo_scale=1.0` 时，不同模型、不同 `request_rate` 下真实运行和模拟运行的 SLO 误差，结果放置在 `./result/slo/backend/plots/llama_xB_slo_subplots.png` 路径下。

4. **`./4-plot_slo_scale_all_run.sh`**  
   用于绘制不同 `slo_scale` 下，不同模型、不同 `request_rate` 下的结果，结果放置在 `./result/slo_scale_plots` 路径下。

5. **`./5-plot_slo_comparison_heatmaps_run.sh`**  
   用于将上一步的结果绘制为热力图，以便更好地展示，结果放置在 `./result/slo/backend/plots/heatmaps` 路径下。

6. **`./6-plot_backend_slo_mean_delta_run.sh`**  
   用于展示不同模型的聚合结果，以便在论文中展示，结果放置在 `./result/slo/fig_5_1a(b).png` 路径下。