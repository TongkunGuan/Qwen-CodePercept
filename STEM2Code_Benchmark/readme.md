
# STEM2Code 评测说明

## 1. 数据下载

### Benchmark JSONL 文件
[点击下载 JSONL 文件](https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/zhongfan_intern/guantongkun/benchmark/STEM2Code20260405.jsonl?OSSAccessKeyId=LTAI5tG6PFYfzqBsgGffVXy3&Expires=2090761443&Signature=%2BudGKeVrhDPVSkEBCPlsA5ApFBY%3D)

### Benchmark 图片文件
[点击下载图片压缩包](https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/zhongfan_intern/guantongkun/benchmark/STEM2Code_benchmark.zip?OSSAccessKeyId=LTAI5tG6PFYfzqBsgGffVXy3&Expires=2090761521&Signature=EOwW7t8PpB7qqU2s6MaZyikVTXo%3D)

---

## 2. 推理环境准备

评测推理依赖仓库 **ms-swift**，请下载并安装：

- GitHub 仓库：  
  https://github.com/modelscope/ms-swift/tree/v3.9.2

### 安装方式

#### 方式一：直接使用官方镜像（推荐）
可参考文档：  
https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html#id2

#### 方式二：手动安装
可参考文档：  
https://swift.readthedocs.io/zh-cn/v3.9/GetStarted/SWIFT%E5%AE%89%E8%A3%85.html

---

## 3. 部署待评测模型

请先部署需要评测的模型，并确保能够通过 OpenAI 兼容接口正常访问。

### 3.1 本地模型部署

如果评测的是本地模型，需要配置以下服务：

- `infer_server_32B_model`
- `infer_server_4B&8B_models`

并获取对应的：

- `api_key`
- `api_base`

建议先在本地完成调试，确认模型能够正常生成结果。

### 3.2 调试示例

```python
from openai import OpenAI

client = OpenAI(
    api_key='none',
    base_url="http://xxx:8000/v1",
    timeout=300,
)

model_name = "CodePercept-32B-image2code-RL"

usr_prompt = (
    "You are an expert Python developer who specializes in writing matplotlib "
    "code based on a given picture. Now, please give me the matplotlib code "
    "that reproduces the picture below."
)

try:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": usr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/zhongfan_intern/guantongkun/remove_mathv_random/images/MathVision_images_1007_v5.png?OSSAccessKeyId=LTAI5tG6PFYfzqBsgGffVXy3&Expires=1774449109&Signature=ZCgyGySIZADg1NtVzfZsva2OIHM%3D"
                        }
                    }
                ],
            }
        ],
        temperature=0.7,
        max_tokens=16384,
        stream=False  # 调试阶段建议关闭流式输出
    )
    print("Response:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(e)
```

---

## 4. 其他模型接入

如果评测的是其他模型，只需要提供可访问的接口信息即可：

- `api_key`
- `api_base`

例如：

- `qwen3.5-plus`

---

## 5. 运行评测

### 第一步：运行 `evaluation.py`

执行后将会生成：

- 一个结果文件夹，例如：`qwen3.5-plus/`
- 一个结果文件，例如：`qwen3.5-plus.jsonl`

### 第二步：运行 `calculate.py`

执行后可得到该模型的最终评测分数。

---

## 6. 评测流程总结

完整流程如下：

1. 下载 benchmark 数据（JSONL + 图片）
2. 安装并配置 `ms-swift` 环境
3. 部署待评测模型，并获取 `api_key` 与 `api_base`
4. 运行 `evaluation.py` 生成模型输出结果
5. 运行 `calculate.py` 计算最终得分
