import json
import argparse
import os

def calculate_final_score(jsonl_file_path: str):
    """
    读取指定的JSONL结果文件，计算并打印包含四个核心指标的详细分数报告。
    """
    # 1. 健壮性检查：确保文件存在
    if not os.path.exists(jsonl_file_path):
        print(f"[错误] 找不到指定的文件: {jsonl_file_path}")
        print("请检查文件名和路径是否正确。")
        return

    # 2. 初始化所有需要的计数器
    total_image_score = 0.0
    total_code_score = 0.0
    successful_records = 0
    total_lines = 0
    
    print(f"\n 正在分析文件: {os.path.basename(jsonl_file_path)} ...")

    # 3. 逐行读取和处理JSONL文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                
                # [核心逻辑] 只处理状态为SUCCESS且分数有效的记录
                if (data.get('status') == 'SUCCESS' and 
                    isinstance(data.get('code_score'), (int, float)) and
                    isinstance(data.get('image_score'), (int, float))):
                    
                    total_image_score += data['image_score']
                    total_code_score += data['code_score']
                    successful_records += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  警告: 第 {line_num} 行解析失败或格式错误，已跳过。错误: {e}")

    # 4. 计算平均分数
    #    (处理 successful_records 为 0 的边缘情况)
    if successful_records > 0:
        avg_image_score = total_image_score / successful_records
        avg_code_score = total_code_score / successful_records
        average_score = (avg_image_score + avg_code_score) / 2
    else:
        avg_image_score = 0.0
        avg_code_score = 0.0
        average_score = 0.0

    # 5. [核心逻辑] 计算执行通过率
    #    - 获取与.jsonl文件同名的图片目录
    base_dir = os.path.dirname(jsonl_file_path)
    model_name = os.path.splitext(os.path.basename(jsonl_file_path))[0]
    image_dir_path = os.path.join(base_dir, model_name)
    
    num_generated_images = 0
    if os.path.isdir(image_dir_path):
        # 只计算文件，忽略可能存在的子目录
        num_generated_images = len([
            name for name in os.listdir(image_dir_path) 
            if os.path.isfile(os.path.join(image_dir_path, name))
        ])
    else:
        print(f"  - 警告: 找不到对应的图片目录: {image_dir_path}，执行率将为0。")
    
    # 根据你的要求，分母固定为1000
    TOTAL_SAMPLES = 1000
    execution_pass_rate = (num_generated_images / TOTAL_SAMPLES) * 100
    final_average_score = (avg_image_score + avg_code_score + execution_pass_rate)*1/3
    # 6. 打印最终的专业报告
    print("\n==================================================")
    print("             模型综合评估报告             ")
    print("==================================================")
    print(f" 模型 (文件):    {model_name}.jsonl")
    print("--------------------------------------------------")
    print(" 核心指标:")
    print(f"   -   平均图像分数:  {avg_image_score:>6.2f}")
    print(f"   -   平均代码分数:  {avg_code_score:>6.2f}")
    print(f"   -   执行通过率:    {execution_pass_rate:>6.2f}%  ({num_generated_images}/{TOTAL_SAMPLES})")
    print("--------------------------------------------------")
    print(f"   -   最终均分:      {final_average_score:>6.2f}")
    print("==================================================")
    print(" 数据统计:")
    print(f"   - 成功评分记录数: {successful_records}")
    print(f"   - 文件总行数:     {total_lines}")
    print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从模型评估结果(.jsonl)文件中计算详细分数报告。",
        formatter_class=argparse.RawTextHelpFormatter, # 优化帮助信息的显示
        epilog=(
            "示例:\n"
            "  python calculate_scores.py ./evaluate_benchmark/gemini_2.5_pro.jsonl\n"
            "  python calculate_scores.py ./evaluate_benchmark/GPT5_high.jsonl"
        )
    )
    parser.add_argument(
        "jsonl_file", 
        help="需要计算分数的 .jsonl 结果文件的路径"
    )
    
    args = parser.parse_args()
    calculate_final_score(args.jsonl_file)
