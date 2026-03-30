import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parse the log file
def parse_log(file_path):
    data = []
    current_model = None
    current_tp = None
    current_stage = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Match model and tp
            model_match = re.search(r'Fitting model (.+?) with tp_world_size (\d+) \((.+?)\)', line)
            if model_match:
                current_model = model_match.group(1)
                current_tp = int(model_match.group(2))
                current_stage = model_match.group(3)
                continue
            
            # Match data lines
            data_match = re.search(r'\s*(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+([-+]?\d+\.\d+)%', line)
            if data_match and current_model and current_tp and current_stage:
                bs = int(data_match.group(1))
                ilen = int(data_match.group(2))
                actual = float(data_match.group(3))
                pred = float(data_match.group(4))
                rel_err = float(data_match.group(5))
                
                # Calculate product of batch size and input length
                bs_ilen_product = bs * ilen
                
                data.append({
                    'model': current_model,
                    'tp': current_tp,
                    'stage': current_stage,
                    'bs': bs,
                    'ilen': ilen,
                    'bs_ilen_product': bs_ilen_product,
                    'actual': actual,
                    'pred': pred,
                    'rel_err': rel_err
                })
    
    return pd.DataFrame(data)

# Generate combined plots
def generate_combined_plots(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get unique models and tps
    models = list(df['model'].unique())  # 转列表
    print(models)
    models.remove('/users/rh/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf')     # 正常删除
    tps = df['tp'].unique()
    stages = df['stage'].unique()

    # Simplify model names for plotting
    model_name_map = {}
    for model in models:
        name = model.split('/')[-1].replace('-', '_')
        # Truncate long names
        if len(name) > 15:
            name = name[:15] + '...'
        model_name_map[model] = name
    
    # 1. Combined plot for all models - Prefill vs Decoding stages
    plt.figure(figsize=(24, 12))
    
    # Prefill stage
    plt.subplot(2, 3, 1)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Prefill stage')]
        for tp in tps:
            tp_df = model_df[model_df['tp'] == tp]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', 
                         label=f'{model_name_map[model]} (TP={tp})')
    plt.xlabel('Batch Size × Input Length')
    plt.ylabel('Time (ms)')
    plt.title('Prefill Stage: Time vs Batch Size × Input Length')
    plt.legend()
    plt.grid(True)
    
    # Decoding stage, small batch size
    plt.subplot(2, 3, 2)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, small batch size')]
        for tp in tps:
            tp_df = model_df[model_df['tp'] == tp]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', 
                         label=f'{model_name_map[model]} (TP={tp})')
    plt.xlabel('Batch Size × Input Length')
    plt.ylabel('Time (ms)')
    plt.title('Decoding Stage (Small Batch): Time vs Batch Size × Input Length')
    plt.legend()
    plt.grid(True)
    
    # Decoding stage, large batch size
    plt.subplot(2, 3, 3)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, large batch size')]
        for tp in tps:
            tp_df = model_df[model_df['tp'] == tp]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', 
                         label=f'{model_name_map[model]} (TP={tp})')
    plt.xlabel('Batch Size × Input Length')
    plt.ylabel('Time (ms)')
    plt.title('Decoding Stage (Large Batch): Time vs Batch Size × Input Length')
    plt.legend()
    plt.grid(True)
    
    # Prefill relative error
    plt.subplot(2, 3, 4)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Prefill stage')]
        for tp in tps:
            tp_df = model_df[model_df['tp'] == tp]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['rel_err'], marker='o', 
                         label=f'{model_name_map[model]} (TP={tp})')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Batch Size × Input Length')
    plt.ylabel('Relative Error (%)')
    plt.title('Prefill Stage: Relative Error vs Batch Size × Input Length')
    plt.legend()
    plt.grid(True)
    
    # Decoding relative error - small batch
    plt.subplot(2, 3, 5)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, small batch size')]
        for tp in tps:
            tp_df = model_df[model_df['tp'] == tp]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['rel_err'], marker='o', 
                         label=f'{model_name_map[model]} (TP={tp})')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Batch Size × Input Length')
    plt.ylabel('Relative Error (%)')
    plt.title('Decoding Stage (Small Batch): Relative Error vs Batch Size × Input Length')
    plt.legend()
    plt.grid(True)
    
    # Decoding relative error - large batch
    plt.subplot(2, 3, 6)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, large batch size')]
        for tp in tps:
            tp_df = model_df[model_df['tp'] == tp]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['rel_err'], marker='o', 
                         label=f'{model_name_map[model]} (TP={tp})')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Batch Size × Input Length')
    plt.ylabel('Relative Error (%)')
    plt.title('Decoding Stage (Large Batch): Relative Error vs Batch Size × Input Length')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_all_models.png')
    plt.close()
    
    # 2. TP comparison for each model
    for model in models:
        model_df = df[df['model'] == model]
        model_name = model_name_map[model]
        
        plt.figure(figsize=(18, 18))
        
        # Prefill: TP=1 vs TP=2
        plt.subplot(3, 2, 1)
        for tp in tps:
            tp_df = model_df[(model_df['tp'] == tp) & (model_df['stage'] == 'Prefill stage')]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', label=f'TP={tp}')
        plt.xlabel('Batch Size × Input Length')
        plt.ylabel('Time (ms)')
        plt.title(f'{model_name}: Prefill Stage - TP Comparison')
        plt.legend()
        plt.grid(True)
        
        # Decoding (Small Batch): TP=1 vs TP=2
        plt.subplot(3, 2, 2)
        for tp in tps:
            tp_df = model_df[(model_df['tp'] == tp) & (model_df['stage'] == 'Decoding stage, small batch size')]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', label=f'TP={tp}')
        plt.xlabel('Batch Size × Input Length')
        plt.ylabel('Time (ms)')
        plt.title(f'{model_name}: Decoding Stage (Small Batch) - TP Comparison')
        plt.legend()
        plt.grid(True)
        
        # Decoding (Large Batch): TP=1 vs TP=2
        plt.subplot(3, 2, 3)
        for tp in tps:
            tp_df = model_df[(model_df['tp'] == tp) & (model_df['stage'] == 'Decoding stage, large batch size')]
            if not tp_df.empty:
                # Group by product of batch size and input length, only for numeric columns
                numeric_columns = ['actual', 'pred', 'rel_err']
                grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
                plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', label=f'TP={tp}')
        plt.xlabel('Batch Size × Input Length')
        plt.ylabel('Time (ms)')
        plt.title(f'{model_name}: Decoding Stage (Large Batch) - TP Comparison')
        plt.legend()
        plt.grid(True)
        
        # Prefill: Actual vs Predicted
        plt.subplot(3, 2, 4)
        tp_df = model_df[(model_df['tp'] == 1) & (model_df['stage'] == 'Prefill stage')]
        if not tp_df.empty:
            # Group by product of batch size and input length, only for numeric columns
            numeric_columns = ['actual', 'pred', 'rel_err']
            grouped = tp_df.groupby('bs_ilen_product')[numeric_columns].mean().reset_index()
            plt.plot(grouped['bs_ilen_product'], grouped['actual'], marker='o', label='Actual')
            plt.plot(grouped['bs_ilen_product'], grouped['pred'], marker='x', label='Predicted')
        plt.xlabel('Batch Size × Input Length')
        plt.ylabel('Time (ms)')
        plt.title(f'{model_name}: Prefill Stage - Actual vs Predicted (TP=1)')
        plt.legend()
        plt.grid(True)
        
        # Decoding (Small Batch): Actual vs Predicted
        plt.subplot(3, 2, 5)
        tp_df = model_df[(model_df['tp'] == 1) & (model_df['stage'] == 'Decoding stage, small batch size')]
        if not tp_df.empty:
            # Group by product of batch size and input length, only for numeric columns
            numeric_columns = ['actual', 'pred', 'rel_err']
            grouped = tp_df.groupby('bs')[numeric_columns].mean().reset_index()
            plt.plot(grouped['bs'], grouped['actual'], marker='o', label='Actual')
            plt.plot(grouped['bs'], grouped['pred'], marker='x', label='Predicted')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (ms)')
        plt.title(f'{model_name}: Decoding Stage (Small Batch) - Actual vs Predicted (TP=1)')
        plt.legend()
        plt.grid(True)
        
        # Decoding (Large Batch): Actual vs Predicted
        plt.subplot(3, 2, 6)
        tp_df = model_df[(model_df['tp'] == 2) & (model_df['stage'] == 'Decoding stage, large batch size')]
        if not tp_df.empty:
            # Group by product of batch size and input length, only for numeric columns
            numeric_columns = ['actual', 'pred', 'rel_err']
            grouped = tp_df.groupby('bs')[numeric_columns].mean().reset_index()
            plt.plot(grouped['bs'], grouped['actual'], marker='o', label='Actual')
            plt.plot(grouped['bs'], grouped['pred'], marker='x', label='Predicted')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (ms)')
        plt.title(f'{model_name}: Decoding Stage (Large Batch) - Actual vs Predicted (TP=2)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_tp_comparison.png')
        plt.close()
    
    # 3. Batch size impact for different models
    plt.figure(figsize=(18, 18))
    
    # Prefill stage - batch size impact
    plt.subplot(3, 2, 1)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Prefill stage') & (df['tp'] == 1)]
        if not model_df.empty:
            # Take a specific input length (e.g., 128)
            ilen_df = model_df[model_df['ilen'] == 64]
            if not ilen_df.empty:
                plt.plot(ilen_df['bs'], ilen_df['actual'], marker='o', 
                         label=f'{model_name_map[model]} (ilen=64)')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title('Prefill Stage: Time vs Batch Size (ilen=64, TP=1)')
    plt.legend()
    plt.grid(True)
    
    # Decoding stage (Small Batch) - batch size impact
    plt.subplot(3, 2, 2)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, small batch size') & (df['tp'] == 1)]
        if not model_df.empty:
            # Take a specific input length (e.g., 128)
            ilen_df = model_df[model_df['ilen'] == 64]
            if not ilen_df.empty:
                plt.plot(ilen_df['bs'], ilen_df['actual'], marker='o', 
                         label=f'{model_name_map[model]} (ilen=64)')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title('Decoding Stage (Small Batch): Time vs Batch Size (ilen=64, TP=1)')
    plt.legend()
    plt.grid(True)
    
    # Decoding stage (Large Batch) - batch size impact
    plt.subplot(3, 2, 3)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, large batch size') & (df['tp'] == 1)]
        if not model_df.empty:
            # Take a specific input length (e.g., 128)
            ilen_df = model_df[model_df['ilen'] == 64]
            if not ilen_df.empty:
                plt.plot(ilen_df['bs'], ilen_df['actual'], marker='o', 
                         label=f'{model_name_map[model]} (ilen=64)')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title('Decoding Stage (Large Batch): Time vs Batch Size (ilen=64, TP=1)')
    plt.legend()
    plt.grid(True)
    
    # Model comparison - Prefill
    plt.subplot(3, 2, 4)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Prefill stage') & (df['tp'] == 1) & (df['bs'] == 16)]
        if not model_df.empty:
            plt.plot(model_df['ilen'], model_df['actual'], marker='o', 
                     label=f'{model_name_map[model]} (bs=16)')
    plt.xlabel('Input Length')
    plt.ylabel('Time (ms)')
    plt.title('Model Comparison: Prefill Stage (bs=16, TP=1)')
    plt.legend()
    plt.grid(True)
    
    # Model comparison - Decoding (Small Batch)
    plt.subplot(3, 2, 5)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, small batch size') & (df['tp'] == 1) & (df['bs'] == 16)]
        if not model_df.empty:
            plt.plot(model_df['ilen'], model_df['actual'], marker='o', 
                     label=f'{model_name_map[model]} (bs=16)')
    plt.xlabel('Input Length')
    plt.ylabel('Time (ms)')
    plt.title('Model Comparison: Decoding Stage (Small Batch) (bs=16, TP=1)')
    plt.legend()
    plt.grid(True)
    
    # Model comparison - Decoding (Large Batch)
    plt.subplot(3, 2, 6)
    for model in models:
        model_df = df[(df['model'] == model) & (df['stage'] == 'Decoding stage, large batch size') & (df['tp'] == 1) & (df['bs'] == 16)]
        if not model_df.empty:
            plt.plot(model_df['ilen'], model_df['actual'], marker='o', 
                     label=f'{model_name_map[model]} (bs=16)')
    plt.xlabel('Input Length')
    plt.ylabel('Time (ms)')
    plt.title('Model Comparison: Decoding Stage (Large Batch) (bs=16, TP=1)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_size_impact.png')
    plt.close()

# Main function
def main():
    log_path = '/users/rh/DistServe/evaluation/1-fit-time-usage/main.log'
    output_dir = '/users/rh/DistServe/evaluation/1-fit-time-usage/plots_combined'
    
    # Parse the log file
    df = parse_log(log_path)
    
    # Generate combined plots
    generate_combined_plots(df, output_dir)
    
    print(f"Combined plots generated successfully in {output_dir}")

if __name__ == '__main__':
    main()
