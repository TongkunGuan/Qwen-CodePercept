#!/bin/bash

# ==================== 默认可配置参数 ====================
# 这些值将在未通过命令行参数指定时使用
MODEL_PATH="/mnt/cpfs_m6_29eu38p1/Group-m6/guantongkun.gtk/weight/sft/caption_qa_8B_20260224/v9-20260306-162000/checkpoint-102500"
START_PORT=10000
SERVED_MODEL_NAME="CodePercept-8B-Caption_QA_PAMI"

# ==================== 命令行参数解析 ====================
# 函数：显示用法说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model-path       <path>     模型路径 (默认: ${MODEL_PATH})"
    echo "  -p, --start-port       <port>     服务起始端口 (默认: ${START_PORT})"
    echo "  -n, --model-name       <name>     服务的模型名称 (默认: ${SERVED_MODEL_NAME})"
    echo "  -h, --help                        显示此帮助信息"
    echo ""
    echo "示例: $0 --model-path /path/to/your/model --start-port 9000 --model-name MyAwesomeModel"
    exit 1
}

# 使用 while 循环和 case 语句来解析参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            if [ -n "$2" ] && [[ $2 != -* ]]; then
                MODEL_PATH="$2"
                shift 2
            else
                echo "[错误] --model-path 需要一个参数值。" >&2
                usage
            fi
            ;;
        -p|--start-port)
            if [ -n "$2" ] && [[ $2 != -* ]]; then
                START_PORT="$2"
                shift 2
            else
                echo "[错误] --start-port 需要一个参数值。" >&2
                usage
            fi
            ;;
        -n|--model-name)
            if [ -n "$2" ] && [[ $2 != -* ]]; then
                SERVED_MODEL_NAME="$2"
                shift 2
            else
                echo "[错误] --model-name 需要一个参数值。" >&2
                usage
            fi
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1" >&2
            usage
            ;;
    esac
done

# ==================== 脚本主体开始 ====================
echo "--- 开始部署 Swift Rollout 服务 ---"
echo ""
echo "--- 使用以下配置 ---"
echo "模型路径 (MODEL_PATH)       : ${MODEL_PATH}"
echo "服务起始端口 (START_PORT)     : ${START_PORT}"
echo "服务模型名称 (SERVED_MODEL_NAME): ${SERVED_MODEL_NAME}"
echo "----------------------"
echo ""


# --- 仅使用 'ip addr' 来获取内部IP地址 ---
INTERNAL_IP=$(ip -4 addr show eth0 | grep -oP 'inet \K[\d.]+')

if [ -z "$INTERNAL_IP" ]; then
    echo "[警告] 未能从'eth0'接口自动检测到IP地址。将尝试查找所有接口..."
    # 尝试从所有接口中查找私有IP
    INTERNAL_IP=$(ip -4 addr | grep -oP 'inet (10(\.\d{1,3}){3}|172\.(1[6-9]|2[0-9]|3[0-1])(\.\d{1,3}){2}|192\.168(\.\d{1,3}){2})' | awk '{print $2}' | head -n 1)
fi

if [ -z "$INTERNAL_IP" ]; then
    echo "[错误] 无法自动获取本机的内部IP地址。请手动运行 'ip addr' 命令查找，并配置客户端。"
    # 脚本可以继续运行，但需要用户手动处理IP问题
else
    echo "[成功] 检测到本机(服务器)的内部IP地址是: $INTERNAL_IP"
    echo "[重要信息] 服务启动后，其他DLC实例应访问: http://$INTERNAL_IP:<PORT>"
fi

echo "[提醒] 请确保防火墙或云服务商安全组规则允许其他实例访问本机的端口范围。"
echo ""


# ==================== swift deploy 相关参数 ====================
# 这些参数将应用于每个启动的服务
export MAX_PIXELS=1310720
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12

# ==================== 启动脚本 ====================
# 循环8次，对应8个GPU (0到7)
for i in $(seq 0 7)
do
  # 计算当前服务要使用的端口号
  PORT=$((START_PORT + 10*i))
  
  # 设置当前服务可见的GPU
  export CUDA_VISIBLE_DEVICES=$i
  
  echo "Starting swift deploy server on GPU $i at port $PORT ..."
  
  # 在后台启动 swift deploy 服务
  # 使用 nohup 和 & 确保即使关闭终端，服务也能继续运行
  # 将日志输出到不同的文件，方便排查问题
  nohup swift deploy \
    --model "${MODEL_PATH}" \
    --port ${PORT} \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 32768 \
    --max_new_tokens 16384 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --host 0.0.0.0 \
    --served_model_name "${SERVED_MODEL_NAME}" > swift_deploy_${SERVED_MODEL_NAME}_gpu${i}.log 2>&1 &
  
  # 打印后台进程的PID，方便管理
  echo "Server for GPU $i started with PID $! on port $PORT. Log file: swift_deploy_${SERVED_MODEL_NAME}_gpu${i}.log"
done

echo ""
echo "All 8 swift deploy services have been launched in the background."
echo "You can check their status with 'ps aux | grep \"swift deploy\"' or 'nvidia-smi'."
echo "To stop all services, you can use the command below:"
echo "pkill -f \"swift deploy --model ${MODEL_PATH}\""

# ==================== 核心修改 ====================
# 使用 wait 命令等待所有后台子进程。
# 因为服务进程不会自己退出，所以 wait 会一直阻塞，从而保持脚本和容器的运行。
wait
