data:

[download jsonl file](https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/zhongfan_intern/guantongkun/benchmark/STEM2Code20260405.jsonl?OSSAccessKeyId=LTAI5tG6PFYfzqBsgGffVXy3&Expires=2090761443&Signature=%2BudGKeVrhDPVSkEBCPlsA5ApFBY%3D)

[download image file](https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/zhongfan_intern/guantongkun/benchmark/STEM2Code_benchmark.zip?OSSAccessKeyId=LTAI5tG6PFYfzqBsgGffVXy3&Expires=2090761521&Signature=EOwW7t8PpB7qqU2s6MaZyikVTXo%3D)
1. 部署要评测的模型。  
   - 如果是本地模型，需要配置 `infer_server_32B_model`、`infer_server_4B&8B_models`，获取 `api_key` 和 `api_base`。  
   - 可先在本地调试，确认能够正常生成。示例如下：

```python
from openai import OpenAI

client = OpenAI(
    api_key='none',
    base_url="http://xxx:8000/v1",
    timeout=300,
)

model_name = "CodePercept-32B-image2code-RL"

usr_prompt = "You are an expert Python developer who specializes in writing matplotlib code based on a given picture. Now, please give me the matplotlib code that reproduces the picture below."

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
        stream=False  # 先不用流式，方便调试
    )
    print("Response:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(e)
```

2. 如果是其他模型，只需提供可访问的 `api_key` 和 `api_base` 即可。  
   - 例如：`qwen3.5-plus`

3. 运行 `evaluation.py` 文件。  
   - 会生成一个文件夹：`qwen3.5-plus`  
   - 同时生成一个结果文件：`qwen3.5-plus.jsonl`

4. 运行 `calculate.py`，得到最终分数。
