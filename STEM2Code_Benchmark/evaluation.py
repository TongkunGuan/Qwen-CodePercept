import os
import re
import io
import sys
import json
import math
import base64
import random
import argparse
import pathlib
import difflib
import traceback # 用于获取更详细的错误信息
from openai import OpenAI
from openai import APIStatusError  
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
# [新增] 导入多进程和函数工具
from multiprocessing import Pool, Manager
from functools import partial
# 使用 'Agg' 后端，这样在服务器上运行时不会尝试打开GUI窗口
import matplotlib
matplotlib.use('Agg')
import subprocess
import tempfile
import textwrap
import requests  # Replaced openai with requests
from PIL import Image
MAX_PIXELS = 1310720  # 16*16*4*1280
def round_by_factor(number: int, factor: int) -> int:
    """ 返回最接近 number 的且能被 factor 整除的整数 """
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """ 返回大于等于 number 的且能被 factor 整除的整数 """
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """ 返回小于等于 number 的且能被 factor 整除的整数 """
    return math.floor(number / factor) * factor
def smart_resize(height, width, factor=32, min_pixels=64*64, max_pixels=16*16*4*1280, max_long_side=8192):
    """ 缩放后图片满足以下条件:
        1. 长宽能被 factor 整除
        2. pixels 总数被限制在 [min_pixels, max_pixels] 内
        3. 最长边限制在 max_long_side 内
        4. 保证其长宽比基本不变
    """
    if height < 2 or width < 2:
        raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f'absolute aspect ratio must be smaller than 100, got {height} / {width}')

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
def extract_gpt_score(resp):
    """
    从GPT的响应文本中提取分数。经过多轮修正，健壮性更强。
    能够处理 "Score: 90/100", "**Score**: 90/100", "Score: 95" 等多种格式。
    并且能避免 "Score: 90/1000" 等错误情况。
    """
    # 主模式：匹配 "xx/100" 格式。在100后添加了单词边界 \b 来确保不会匹配 1000, 1001等。
    pattern = r"[*_~`]*\s*Score\s*[*_~`]*\s*:\s*[*_~`]*(\d{1,3})\s*/\s*100\b"
    m = re.search(pattern, resp, re.IGNORECASE | re.MULTILINE)
    if m:
        return int(m.group(1))

    # 备用模式：匹配 "Score: 95" 这种不含 "/100" 的格式。
    # 使用了负向先行断言 (?!\s*/)：确保数字后面不跟着斜杠。
    # 使用了单词边界 \b：确保我们匹配的是一个完整的数字 (e.g., 95, not 9 from 95)。
    fallback_pattern = r"[*_~`]*\s*Score\s*[*_~`]*\s*:\s*[*_~`]*(\d{1,3})\b(?!\s*/)"
    matches = list(re.finditer(fallback_pattern, resp, re.IGNORECASE))
    if matches:
        # 取最后一个匹配项
        return int(matches[-1].group(1))

    # 如果都无法匹配
    print(f"⚠️ 无法从响应中提取GPT分数: {resp}")
    # import pdb; pdb.set_trace()
    return None

def image_to_data_uri(image_path: Path) -> Optional[str]:
    """
    读取图片文件，将其编码为Base64，并格式化为Data URI。(此函数保持不变)
    """
    suffix = image_path.suffix.lower()
    mime_type_map = {'.jpg': 'image/jpeg', '.png': 'image/png', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif'}
    mime_type = mime_type_map.get(suffix)
    if not mime_type:
        # print(f"⚠️ 跳过不支持的文件类型: {image_path.name}")
        return None
    try:
        img = Image.open(image_path)
        if img.width * img.height > MAX_PIXELS:
            height, width = smart_resize(img.height, img.width)
            img = img.resize((width, height))
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # with open(image_path, "rb") as image_file:
        #     base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except IOError as e:
        # print(f"❌ 读取文件时出错: {image_path}. 错误: {e}")
        return None

def extract_python_code(response_text: str) -> Optional[str]:
    """
    从包含Markdown代码块的文本中提取Python代码。(此函数保持不变)
    """
    if not response_text:
        return None
    # match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    # if match:
    #     return match.group(1).strip()
    print(f"正在提取Python代码块: {response_text}")
    code_blocks = re.findall(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if len(code_blocks) > 0:
        if 'doubao' in args.name or 'MiMo' in args.name or "Intern-S1" in args.name:
            return code_blocks[-1].strip()
        return code_blocks[0].strip()
    # match = re.search(r"```\n(.*?)\n```", response_text, re.DOTALL)
    # if match:
    #     return match.group(1).strip()
    code_blocks = re.findall(r"```\n(.*?)\n```", response_text, re.DOTALL)
    if len(code_blocks) > 0:
        if 'doubao' in args.name or 'MiMo' in args.name or "Intern-S1" in args.name:
            return code_blocks[-1].strip()
        return code_blocks[0].strip()
    return None

def insert_line_before_show(original_code_string: str, save_path: str) -> str:
    """
    在包含 'plt.show()' 的代码行之前插入 'inspect_matplotlib_ax(ax)'。

    Args:
        original_code_string: 包含原始 Python 代码的字符串。

    Returns:
        修改后的 Python 代码字符串。
    """
    lines = original_code_string.splitlines()
    new_lines = []
    
    inserted = False
    for line in lines:
        # 检查该行是否包含 plt.show()，并处理前后空格
        if 'plt.show()' in line and not inserted:
            # 获取 plt.show() 行的缩进
            indentation = line[:line.find('plt.show()')]
            # 在 plt.show() 之前添加新行，并保持相同的缩进
            new_lines.append(indentation + f"plt.savefig('{save_path}', dpi=300)")
            inserted = True
        else:
            new_lines.append(line)
        
    return '\n'.join(new_lines)

def insert_line_before_show(original_code_string: str, save_path: str) -> str:
    """
    智能地插入或替换 plt.savefig 调用。

    此函数的规则如下：
    1. 遍历代码，寻找第一个出现的 `plt.savefig(...)` 或 `plt.show()`。
    2. 如果首先找到的是 `plt.savefig(...)` 行（无论是否被注释），
       则用新的 `plt.savefig('{save_path}', dpi=300)` **替换**该行。
    3. 如果首先找到的是 `plt.show()` 行，则在该行**之前插入**新的 
       `plt.savefig('{save_path}', dpi=300)`。
    4. 如果代码中既没有 `plt.savefig(...)` 也没有 `plt.show()`，
       则在代码的末尾**追加**新的 `savefig` 命令。
    5. 一旦完成插入或替换，后续的 `savefig` 或 `show` 调用将保持不变。

    Args:
        original_code_string: 包含原始 Python 代码的字符串。
        save_path: 用于保存图像的文件路径。

    Returns:
        修改后的 Python 代码字符串。
    """
    lines = original_code_string.splitlines()
    new_lines = []
    
    # 标记是否已经执行了插入/替换操作
    is_handled = False
    
    for line in lines:
        # 检查是否是我们的目标行，并且我们还没有处理过
        if not is_handled and ('plt.savefig(' in line or 'plt.show()' in line):
            is_handled = True
            
            # 获取当前行的缩进，以便新行保持格式一致
            stripped_line = line.lstrip()
            indentation = line[:len(line) - len(stripped_line)]
            
            # 创建新的 savefig 行
            new_savefig_line = f"{indentation}plt.savefig('{save_path}', dpi=300)"

            # 优先处理 savefig：如果找到它，就替换掉当前行
            if 'plt.savefig(' in line:
                new_lines.append(new_savefig_line)
            # 如果找到的是 show，则先插入新行，再保留原始行
            elif 'plt.show()' in line:
                new_lines.append(new_savefig_line)
                new_lines.append(line)
        else:
            # 如果不是目标行，或者我们已经处理过了，就直接添加原始行
            new_lines.append(line)
            
    # 如果遍历完所有行都没有找到 savefig 或 show，则在末尾追加
    if not is_handled:
        # 在追加前，移除末尾可能存在的空行
        while new_lines and not new_lines[-1].strip():
            new_lines.pop()
        new_lines.append(f"plt.savefig('{save_path}', dpi=300)")

    return '\n'.join(new_lines)


def add_plt(code):
    func = """
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['WenQuanYi Zen Hei', 'SimHei','Segoe UI Emoji', 'Noto Color Emoji'] + plt.rcParams['font.serif']# 推荐
plt.rcParams['font.family'] = ['sans-serif','WenQuanYi Zen Hei','Segoe UI Emoji', 'Noto Color Emoji']
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei','Segoe UI Emoji', 'Noto Color Emoji'] + plt.rcParams['font.sans-serif']\n"""
    code = func + code
    return code
def worker_execute_code(code, save_path):
    original_cwd = os.getcwd()
    # if execte_root: os.chdir(execte_root)
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as temp_script:
            full_code = add_plt(code)
            full_code = insert_line_before_show(full_code, save_path)
            temp_script.write(full_code)
            temp_script_path = temp_script.name
        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30  # 设置一个超时时间，防止代码卡死
        )

        # 检查 stderr 中是否有字体警告
        if "does not have a glyph for" in result.stderr:
            print("\n" + "="*60)
            print("子进程执行时检测到 Matplotlib 字体 Glyph 警告！")
            print("--- Stderr 输出 ---")
            print(result.stderr.strip())
            print("="*60 + "\n")
            # 这里可以进入 pdb，但注意，PDB 将在主进程中，
            # 你无法直接检查子进程的内部变量。但你可以检查 full_code 和 result。
            # import pdb; pdb.set_trace()
        
        # 检查子进程是否因其他错误而失败
        if result.returncode != 0:
            # import pdb; pdb.set_trace()
            error_info = f"--- 子进程执行失败 (Exit Code: {result.returncode}) ---\n"
            error_info += "--- Stdout ---\n" + result.stdout.strip() + "\n"
            error_info += "--- Stderr ---\n" + result.stderr.strip() + "\n"
            error_info += "------------------------------------------------"
            print(error_info)
            # 同样可以在这里进入 PDB 调试
            # import pdb; pdb.set_trace()
            return (False, error_info)

        # 如果一切顺利
        return (True, None)

    except subprocess.TimeoutExpired:
        error_info = "--- 代码执行超时 ---"
        print(error_info)
        return (False, error_info)
    except Exception as e:
        error_info = f"--- 运行子进程时主脚本发生错误 ---\n{traceback.format_exc()}"
        print(error_info)
        return (False, error_info)
    finally:
        # 清理工作
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)  # 手动删除临时文件
        plt.close('all')  # 关闭主进程中可能打开的任何图形
        os.chdir(original_cwd)

def Image_Scoring(args, client, item) -> Optional[str]:
    refer_image = item['GT_Image']
    initial_response = item['initial_response'].replace("```python", '').replace("```", '').strip()
    AI_image = os.path.join(args.output_image_dir, refer_image.split('/')[-1])
    # import pdb; pdb.set_trace()
    SUCCESS, error_info = worker_execute_code(initial_response, AI_image)
    if not SUCCESS or not os.path.exists(AI_image):
        print(error_info)
        return 0
    
    system_prompt = "You are an useful assistant."
    user_prompt = """You are an expert judge in evaluating mathematical and geometric diagrams. The first image (reference image) is a ground truth mathematical figure, and the second image (AI-generated image) is created using code generated by an AI assistant. Your task is to score how well the AI-generated image matches the ground truth image.

### Scoring Methodology:
The AI-generated image's score is based on the following criteria, totaling a score out of 100 points. The evaluation must consider the mathematical and geometric correctness of the figure, focusing on the precise arrangement and relationships of its components.

1.  **Geometric & Structural Completeness (30 points)**
    *   **Element Types:** Does the AI-generated image include all fundamental element types from the reference image (e.g., points, lines, segments, rays, circles, polygons, curves, coordinate axes, text labels)?
    *   **Element Quantity:** Is the **exact number** of each element type correct? (e.g., if the reference has 8 points and 3 triangles, does the generated image also have exactly 8 points and 3 triangles?).

2.  **Positional & Relational Accuracy (30 points)**
    *   **Absolute & Relative Positioning:** Are all elements placed at their correct locations? This assesses accuracy within the image's implicit or explicit coordinate system (e.g., points on a grid, vertices of a polygon, center of a circle).
    *   **Spatial Relationships:** Does the image correctly represent all spatial relationships, such as **adjacency** (shapes touching), **intersection** (lines/shapes crossing), **containment** (one shape inside another), **collinearity** (points on a single line), and **parallelism/perpendicularity**?
    *   **Sequential & Topological Relationships:** For figures like graphs or paths, is the **sequence of connections** correct? Is the overall topological structure (e.g., how regions are connected or separated) preserved?
    *   **Layering (Z-order):** Are overlapping elements stacked in the correct order (e.g., is the shaded region correctly drawn behind the boundary line)?

3.  **Text & Annotation Fidelity (10 points)**
    *   **Content:** Does the AI-generated image include all text and symbolic annotations from the reference (e.g., vertex labels like 'A', 'B', angle measures like '90°', 'α', length labels, function equations)? Is the content of the text identical?
    *   **Positioning & Association:** Are annotations placed correctly relative to the geometric elements they describe? (e.g., is the label 'A' next to the correct vertex? Is the angle marker 'α' in the correct corner?).
    *   **Style:** Does the style of the text (e.g., font, size, italics for variables, use of mathematical symbols) match the reference?

4.  **Visual & Stylistic Consistency (20 points)**
    *   **Colors & Fill:** Do the colors (stroke, fill) of all elements match the reference? Are shaded regions filled correctly?
    *   **Line & Marker Styles:** Do line styles (e.g., solid, dashed, dotted), line weights, and marker styles (e.g., dots, small circles, crosses) match?
    *   **Overall Aesthetics:** Does the overall appearance, including background color, grid lines, and aspect ratio, match the reference image?

5.  **Clarity & Legibility (10 points)**
    *   Is the AI-generated image clear, sharp, and well-rendered?
    *   Are there any distracting artifacts, incorrect overlaps, or elements that are difficult to distinguish? Is all text legible?

### Evaluation:
Compare the reference image and AI-generated image head to head. 
Provide a detailed assessment and score for each criterion, then calculate the final total score.
**You must strictly adhere to the following format for your response.(Highest Priority)**

---

Comments:
- Geometric & Structural Completeness: ${your comment and subscore}
- Positional & Relational Accuracy: ${your comment and subscore}
- Text & Annotation Fidelity: ${your comment and subscore}
- Visual & Stylistic Consistency: ${your comment and subscore}
- Clarity & Legibility: ${your comment and subscore}

Score: ${total score}/100

---

Please use the above format to ensure the evaluation is clear and comprehensive.
"""
    for i in range(args.max_retries):
        try:
            completion = client.chat.completions.create(
                model=args.image_scoring_model,
                messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(pathlib.Path(refer_image))}},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(pathlib.Path(AI_image))}}
                ]}
                ],
                temperature=0,
                n=1,
                stream=False,
                max_tokens=16384
            )
            response_text = completion.choices[0].message.content
            score = extract_gpt_score(response_text)
            # import pdb; pdb.set_trace()
            print(f"✅ 图像打分成功：{score}")
            return score
        except APIStatusError as e:
            print(f"❌ API 调用状态错误 (APIStatusError): {e} | Key: {client.api_key}")
        except Exception as e:
            # import pdb; pdb.set_trace()
            print(f"❌ 图像打分失败：{e} {client.api_key}")
            time.sleep(2 * i)
    return None

def Code_Scoring(args, client, item) -> Optional[str]:
    GT_code = item['GT_Code']
    response_code = item['initial_response'] 
    if response_code == "":
        raise ValueError("响应代码为空")

    system_prompt = "You are an useful assistant."
    user_prompt = """You will act as an expert judge, responsible for rigorous visual verification of AI-generated graphics code.

Your sole task is to evaluate whether the AI ​​code is completely consistent with the reference code written by human experts in terms of the final rendered visual result. You must ignore technical differences in the code implementation (e.g., algorithms, data structures) and focus on every pixel and geometric detail that goes into rendering the final image.

--

### Evaluation Mission and Core Principles

1. Visual Identity: Two pieces of code that render the exact same image should be considered equally valid. Elegance or clumsiness of the implementation is irrelevant to the scoring.
2. Pixel-Level Accuracy: Your evaluation must be accurate down to the pixel level. This includes geometric shape outlines, position, number of elements, relative relationships, and all visual attributes.
3. Objective and Quantitative: All comments must be supported by concrete visual evidence, strictly adhering to the following scoring criteria.
4. Unconditional Evaluation: The evaluation must be performed on any provided AI-generated code, **regardless of whether it is empty or incomplete**. You must be scored accordingly by applying the standard criteria, which will naturally result in a very low score.

--

### Scoring Criteria and Guidelines (out of 100 points)

You will be scored based on the following five criteria. Each item is directly related to the final visual presentation.

**1. Overall Layout and Visual Attribute Fidelity (20 points)**
* **Canvas and Coordinate System**: Are the canvas attributes (e.g., aspect ratio, background color) correct? If a grid or coordinate system exists, are its range, scale, and scale consistent with the reference standard?
* **Macro Layout**: Is the overall basic framework of the graphic correct? (For example, where is the main subject located on the canvas, and is the overall visual center of gravity consistent?)
* **Color and Style**: Are the fill color, stroke color, and opacity of all elements consistent with the reference code? Do the line width, style (solid, dashed, dotted), and cap style (round, square) match?
* **Text and Annotations**: If text labels or mathematical annotations exist, are their content, font, size, position, and alignment consistent with the reference code?

**2. Quantitative Fidelity (20 points)**
* **Element List Verification**: Does the AI-generated graphic contain the exact same types and numbers of geometric elements as the reference code? (e.g., 8 polygons, 14 path nodes, 1 mesh).
* **Completeness**: Are there any missing or redundant geometric components compared to the reference code?

**3. Positioning and Layout Accuracy (30 points)**
* **Absolute Coordinate Accuracy**: Do the coordinates of all key elements (e.g., polygon vertices, circle centers, path anchor points) precisely match those calculated in the reference code?
* **Relative Position Relationship**: Are the spatial arrangement of elements correct? (e.g., A is above and left of B, C and D are horizontally aligned, and a group of elements are arranged in a circular pattern).
* **Alignment and Distribution**: Do the elements follow the same alignment (left/right/center) and distribution (uniform/non-uniform) pattern as the reference code?

**4. Relationship and Stacking Completeness (20 points)**
* **Connectivity and Sequence**: If the graph contains paths, networks, or ordered sequences, is the order of connections between nodes **perfectly reproduced**? Are the starting and ending points of lines or paths correct?
* **Spatial Interaction**: Are complex relationships between elements (such as adjacency, containment, intersection, and overlap) rendered correctly? Are the shapes and sizes of overlapping areas accurate?
* **Stacking Order (Z-index**): When elements overlap, are they stacked in the correct order (i.e., which element is on top and which is on the bottom)?

**5. Code Implementation and Quality (10 points)**
* **Clarity and Readability**: Is the code well-structured and clear? Does it use meaningful variable names and appropriate comments?
* **Correctness and Efficiency**: Is the code free of syntactical errors, logical errors, and unnecessary redundancy? Does it effectively use appropriate functions and methods from relevant libraries?
* **Reproducibility**: When executed in the correct environment, does the code run correctly and produce the expected complete graph? ---

### Evaluation:
Compare the reference code to the AI ​​code.
Provide a detailed evaluation and rating for each criterion, and then calculate a final overall score.
**!!!You must strictly adhere to the following format for your response. (Highest Priority)**
**!!!You must strictly adhere to the following format for your response. (Highest Priority)**
**You must strictly adhere to the following format for your response. (Highest Priority)**

---

Comments:
- **Overall Layout and Visual Attribute Fidelity**: ${your comments and sub-score}
- **Quantitative Fidelity**: ${your comments and sub-score}
- **Positioning and Layout Accuracy**: ${your comments and sub-score}
- **Relationship and Stacking Completeness**: ${your comments and sub-score}
- **Code Implementation and Quality**: ${your comments and sub-score}

**Score**: ${total score}/100
--- 
""" + f"""

Now, give your reference code and AI-generated code in the following format:

```python
# Reference Code
{GT_code}
```
```python
# AI-Generated Code
{response_code}
```
Please use the above format to ensure the evaluation is clear and comprehensive.
"""
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=300,
    )
    for i in range(args.max_retries):
        try:
            completion = client.chat.completions.create(
                model=args.code_scoring_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                n=1,
                stream=False,
                max_tokens=16384,
            )
            response_text = completion.choices[0].message.content
            score = extract_gpt_score(response_text)
            print(f"✅ code打分成功： {score}")
            return score
        except APIStatusError as e:
            print(f"❌ API 调用状态错误 (APIStatusError): {e} | Key: {client.api_key}")
        except Exception as e:
            print(f"❌ code打分失败： {e} {client.api_key}")
            time.sleep(2 * i)
    return None
# --- [新增功能 1] ---
def get_completed_uuids(output_filepath: str) -> Set[str]:
    """
    读取已存在的输出文件，返回所有已成功任务的UUID集合。
    """
    completed_uuids = set()
    if not os.path.exists(output_filepath):
        return completed_uuids
    
    with open(output_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                status = data.get('status')
                # 如果任务成功或调试后成功，则记录其uuid
                if status in ['SUCCESS', 'DEBUGGED_SUCCESS']:
                    completed_uuids.add(data.get('uuid'))
            except json.JSONDecodeError:
                # 文件行可能不完整，忽略错误
                continue
    return completed_uuids

# --- [新增功能 2] ---
def save_result_safely(result: Dict, filepath: str, lock):
    """
    使用文件锁安全地将单条结果追加到JSONL文件中。
    """
    with lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def get_semi_success_items(output_filepath: str) -> Dict[str, Dict]:
    """
    读取已存在的输出文件，返回所有半成功任务的信息。
    """
    semi_success_items = {}
    with open(output_filepath, "r", encoding='utf-8') as f:
        # 为了可复现性，可以考虑不随机打乱，或使用固定的随机种子
        for line in f:
            try:
                line = json.loads(line)
                status = line.get('status')
                # 如果任务成功或调试后成功，则记录其uuid
                if status == 'Scoring FAILED' or status == 'Code Scoring FAILED' or status == 'Image Scoring FAILED':
                    semi_success_items[line.get('uuid')] = line
            except json.JSONDecodeError:
                # 文件行可能不完整，忽略错误
                continue
    return semi_success_items
# --- 主处理函数 ---

def create_api_request_jsonl(
    base_data_path: str,
    completed_uuids: Set[str], # [改动] 接收已完成的UUID集合
    model_name: str = "gemini-2.5-pro",
    json_name: str = 'remove_mathv_random',
    output_path: str = 'output.jsonl',
) -> List[Dict]:
    """
    准备API请求列表，并跳过已经成功处理的任务。
    """
    base_data_p = Path(base_data_path)
    jsonl_path = base_data_p / f"{json_name}.jsonl"
    api_requests = []

    if not jsonl_path.exists():
        print(f"错误: {jsonl_path} 不存在。")
        return []

    with open(jsonl_path, "r", encoding='utf-8') as f:
        # 为了可复现性，可以考虑不随机打乱，或使用固定的随机种子
        jsonl_items = [json.loads(line) for line in f]

    semi_success_items = {}
    if os.path.exists(output_path):
        print("正在处理半成功任务...")
        semi_success_items = get_semi_success_items(output_path)

    print("正在准备API请求，并过滤已完成的任务...")
    for i, jsonl_item in tqdm(enumerate(jsonl_items)):
        # 使用文件路径和索引生成唯一的、可复现的UUID
        uuid = f"{i}"

        # [改动] 如果UUID在已完成集合中，则跳过
        if uuid in completed_uuids:
            continue
        
        # import pdb; pdb.set_trace()
        image_path = jsonl_item['messages'][1]['content'][0]['image']
        code_content = jsonl_item['messages'][3]['content'][0]['text']
        if 'doubao' in args.name:
            question = f"""You are an expert Python developer who specializes in writing matplotlib code based on a given picture. Now, please give me a complete matplotlib code enclosed with ```python ``` that reproduces the picture below."""
        else:
            question = f"""You are an expert Python developer who specializes in writing matplotlib code based on a given picture. Now, please give me the matplotlib code that reproduces the picture below."""
        system_prompt = 'You are an useful assistant.'
        api_format = {
            "uuid": uuid,
            "original_item": jsonl_item,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_path}}
                ]}
            ],
            "params": {
                "model": model_name, "temperature": 1, "n": 1,
            },
            'GT_Code': code_content,
            'GT_Image': image_path,
            'initial_response': semi_success_items.get(uuid, {}).get('initial_response', None),
            'code_score': semi_success_items.get(uuid, {}).get('code_score', None),
            'image_score': semi_success_items.get(uuid, {}).get('image_score', None),

        }
        api_requests.append(api_format)

    return api_requests

def call_api_with_retry_requests(args, client, messages):
    # --- API call using requests ---
    # import pdb; pdb.set_trace()
    api_key = client.api_key
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    paras = get_model_params(args.name)
    payload = dict(
        messages=messages,
        n=1,
        **paras)
    payload.pop('retry')
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=300)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        resp_struct = json.loads(response.text)
        answer = resp_struct['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        print(f"❌代码生成失败：{e} {client.api_key}")
        return None
    return None
def call_api_with_retry_qwen(args, client, messages):
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复
    enable_thinking = True
    stream = False
    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model=args.name,
        messages=messages,
        stream=stream,
        # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
        # qwen3-vl-plus、 qwen3-vl-plus-2025-09-23可通过enable_thinking开启或关闭思考、对于qwen3-vl-235b-a22b-thinking，enable_thinking仅支持开启，其他Qwen-VL模型均不适用
        # extra_body={
        #     'enable_thinking': enable_thinking,
        #     # "thinking_budget": 500
        #     },
        # 解除以下注释会在最后一个chunk返回Token使用量
        # stream_options={
        #     "include_usage": True
        # }
    )
    try:
        if not stream:
            answer_content = completion.choices[0].message.content
        else:
            if enable_thinking:
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
            for chunk in completion:
                # 如果chunk.choices为空，则打印usage
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    # 打印思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content != "" and is_answering is False:
                            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True
                        # 打印回复过程
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content
        return answer_content
    except APIStatusError as e:
        print(f"❌ API 调用状态错误 (APIStatusError): {e} | Key: {client.api_key}")
        return None
    except Exception as e:
        print(f"❌代码生成失败：{e} {client.api_key}")
        return None
def call_api_with_retry(args, client, messages):
    if args.mode == 'requests':
        return call_api_with_retry_requests(args, client, messages)
    if args.mode == 'qwen':
        return call_api_with_retry_qwen(args, client, messages)
    for i in range(args.max_retries):
        try:
            paras = get_model_params(args.name)
            # import pdb; pdb.set_trace()
            paras.pop('retry')
            print('start calling')
            completion = client.chat.completions.create(
                messages=messages,
                n=1,
                stream=True,
                **paras               
            )
            # content = completion.choices[0].message.content
            is_answering = False   # 判断是否结束思考过程并开始回复
            content = ''
            reasoning_content = ''
            for chunk in completion:
                # 如果chunk.choices为空，则打印usage
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    # 打印思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content != "" and is_answering is False:
                            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True
                        # 打印回复过程
                        print(delta.content, end='', flush=True)
                        content += delta.content
            print(content)
            if len(content.split('</think>')) > 1:
                content = content.split('</think>')[1]
            return content
        except Exception as e:
            print(f"❌代码生成失败：{e}. {client.api_key} Retrying... ({i + 1}/{args.max_retries})")
            time.sleep(2 * i)


def call_single(task_with_key, args, output_filepath, lock) -> Dict:
    """
    [核心改造]
    处理单个任务：调用API，执行代码，调试，并使用锁安全地保存结果。
    不再返回Optional，总是返回一个带有状态的字典。
    """
    key, line_dict = task_with_key
    scoring_client = OpenAI(
        api_key=key,
        base_url=args.api_base,
        timeout=300,
    )
    if args.name == '4b_instruct':
        key = "EMPTY"
        base_url = "http://10.71.253.161:22002/v1"
    elif args.name == '8b_instruct':
        key = "EMPTY"
        base_url = "http://10.71.253.24:22002/v1"
    elif args.name == '4b_thinking':
        key = "EMPTY"
        base_url = "http://10.71.253.73:22002/v1"
    elif args.name == '8b_thinking':
        key = "EMPTY"
        base_url = "http://10.71.254.34:22002/v1"
    elif args.name == "qwen3-vl-32b-instruct":
        key = "EMPTY"
        base_url = "http://10.71.255.102:22002/v1"
    elif args.name == "qwen3-vl-32b-thinking":
        key = "EMPTY"
        base_url = "http://22.7.246.134:8000/v1"
    elif args.name == 'Qwen3-VL-8B-Instruct-CaptionCode':
        key = "EMPTY"
        base_url = random.choice(
            [
                "http://0.0.0.0:8000/v1",
                "http://0.0.0.0:8010/v1",
                "http://0.0.0.0:8020/v1",
                "http://0.0.0.0:8030/v1",
                "http://0.0.0.0:8040/v1",
                "http://0.0.0.0:8050/v1",
                "http://0.0.0.0:8060/v1",
                "http://0.0.0.0:8070/v1",
            ]
        )
    elif args.name == "Qwen3-VL-8B-Instruct-CaptionCode-Stage2":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.15.0.216:10000/v1',
            'http://22.15.0.216:10010/v1',
            'http://22.15.0.216:10020/v1',
            'http://22.15.0.216:10030/v1',
            'http://22.15.0.216:10040/v1',
            'http://22.15.0.216:10050/v1',
            'http://22.15.0.216:10060/v1',
            'http://22.15.0.216:10070/v1',
        ])
    elif args.name == "Qwen3-VL-4B-Instruct-CaptionCode":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.8.150.142:10000/v1',
            'http://22.8.150.142:10010/v1',
            'http://22.8.150.142:10020/v1',
            'http://22.8.150.142:10030/v1',
            'http://22.8.150.142:10040/v1',
            'http://22.8.150.142:10050/v1',
            'http://22.8.150.142:10060/v1',
            'http://22.8.150.142:10070/v1',
        ])
    elif args.name == "Qwen3-VL-235BA22-Instruct-CaptionCode":
        key = "EMPTY"
        base_url = random.choice([
            'http://10.71.254.106:22002/v1',
        ])
    elif args.name == "Qwen3-VL-32B-Instruct-CaptionCode-noS":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.3.237.240:8000/v1',
        ])
    elif args.name == "Qwen3-VL-4B-Instruct-Code":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.5.97.210:10000/v1',
            'http://22.5.97.210:10010/v1',
            'http://22.5.97.210:10020/v1',
            'http://22.5.97.210:10030/v1',
            'http://22.5.97.210:10040/v1',
            'http://22.5.97.210:10050/v1',
            'http://22.5.97.210:10060/v1',
            'http://22.5.97.210:10070/v1',
        ])
    elif args.name == "Qwen3-VL-8B-Instruct-Code":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.5.107.133:10000/v1',
            'http://22.5.107.133:10010/v1',
            'http://22.5.107.133:10020/v1',
            'http://22.5.107.133:10030/v1',
            'http://22.5.107.133:10040/v1',
            'http://22.5.107.133:10050/v1',
            'http://22.5.107.133:10060/v1',
            'http://22.5.107.133:10070/v1',
        ]) 
    elif args.name == "CodePercept-R1-60":
        key = "EMPTY"
        base_url = 'http://22.14.120.23:8000/v1'
    elif args.name == "CodePercept-R1-300":
        key = "EMPTY"
        base_url = 'http://22.6.222.231:8000/v1'
    elif args.name == "CodePercept-S1-Code-Grpo-30":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.3.239.212:10000/v1',
            'http://22.3.239.212:10010/v1',
            'http://22.3.239.212:10020/v1',
            'http://22.3.239.212:10030/v1',
            'http://22.3.239.212:10040/v1',
            'http://22.3.239.212:10050/v1',
            'http://22.3.239.212:10060/v1',
            'http://22.3.239.212:10070/v1',
        ]) 
    elif args.name == "CodePercept-S1-Code-Grpo-90":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.6.221.160:10000/v1',
            'http://22.6.221.160:10010/v1',
            'http://22.6.221.160:10020/v1',
            'http://22.6.221.160:10030/v1',
            'http://22.6.221.160:10040/v1',
            'http://22.6.221.160:10050/v1',
            'http://22.6.221.160:10060/v1',
            'http://22.6.221.160:10070/v1',
        ]) 
    elif args.name == "CodePercept-S1-Code-Grpo-120":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.5.239.52:10000/v1',
            'http://22.5.239.52:10010/v1',
            'http://22.5.239.52:10020/v1',
            'http://22.5.239.52:10030/v1',
            'http://22.5.239.52:10040/v1',
            'http://22.5.239.52:10050/v1',
            'http://22.5.239.52:10060/v1',
            'http://22.5.239.52:10070/v1',
        ]) 
    elif args.name == "CodePercept-S1-Code-Grpo-150":
        key = "EMPTY"
        base_url = random.choice([
            'http://22.7.240.134:10000/v1',
            'http://22.7.240.134:10010/v1',
            'http://22.7.240.134:10020/v1',
            'http://22.7.240.134:10030/v1',
            'http://22.7.240.134:10040/v1',
            'http://22.7.240.134:10050/v1',
            'http://22.7.240.134:10060/v1',
            'http://22.7.240.134:10070/v1',
        ]) 
    elif args.name == "MiMo-VL-7B-RL":
        key = "sk-123456"
        base_url = random.choice([
            'http://22.14.123.11:10000/v1',
            'http://22.14.123.11:10010/v1',
            'http://22.14.123.11:10020/v1',
            'http://22.14.123.11:10030/v1',
            'http://22.14.123.11:10040/v1',
            'http://22.14.123.11:10050/v1',
            'http://22.14.123.11:10060/v1',
            'http://22.14.123.11:10070/v1',
        ])
    elif args.name == "Keye-VL-1_5-8B":
        key = "sk-123456"
        base_url = random.choice([
            'http://22.6.208.10:10000/v1',
            'http://22.6.208.10:10010/v1',
            'http://22.6.208.10:10020/v1',
            'http://22.6.208.10:10030/v1',
            'http://22.6.208.10:10040/v1',
            'http://22.6.208.10:10050/v1',
            'http://22.6.208.10:10060/v1',
            'http://22.6.208.10:10070/v1',
        ])
    elif args.name == "Ovis2.5-9B":
        key = "sk-123456"
        base_url = random.choice([
            'http://22.5.153.26:10000/v1',
            'http://22.5.153.26:10010/v1',
            'http://22.5.153.26:10020/v1',
            'http://22.5.153.26:10030/v1',
            'http://22.5.153.26:10040/v1',
            'http://22.5.153.26:10050/v1',
            'http://22.5.153.26:10060/v1',
            'http://22.5.153.26:10070/v1',
        ])     
    elif args.name == "InternVL3_5-8B":   
        key = "sk-123456"
        base_url = random.choice([
            'http://22.5.104.160:10000/v1',
            'http://22.5.104.160:10010/v1',
            'http://22.5.104.160:10020/v1',
            'http://22.5.104.160:10030/v1',
            'http://22.5.104.160:10040/v1',
            'http://22.5.104.160:10050/v1',
            'http://22.5.104.160:10060/v1',
            'http://22.5.104.160:10070/v1',
        ])   
    elif args.name == "GLM-4.1V-9B-Base":
        key = "sk-123456"
        base_url = random.choice([
            'http://22.5.108.245:10000/v1',
            'http://22.5.108.245:10010/v1',
            'http://22.5.108.245:10020/v1',
            'http://22.5.108.245:10030/v1',
            'http://22.5.108.245:10040/v1',
            'http://22.5.108.245:10050/v1',
            'http://22.5.108.245:10060/v1',
            'http://22.5.108.245:10070/v1',
        ])
    elif args.name == "Intern-S1-mini":   
        key = "sk-123456"
        base_url = random.choice([
            'http://22.7.249.252:10000/v1',
            'http://22.7.249.252:10010/v1',
            'http://22.7.249.252:10020/v1',
            'http://22.7.249.252:10030/v1',
            'http://22.7.249.252:10040/v1',
            'http://22.7.249.252:10050/v1',
            'http://22.7.249.252:10060/v1',
            'http://22.7.249.252:10070/v1',
        ])
    elif args.name == "MiniCPM-V-4_5":
        key = "sk-123456"
        base_url = random.choice([
            'http://22.7.240.133:10000/v1',
            'http://22.7.240.133:10010/v1',
            'http://22.7.240.133:10020/v1',
            'http://22.7.240.133:10030/v1',
            'http://22.7.240.133:10040/v1',
            'http://22.7.240.133:10050/v1',
            'http://22.7.240.133:10060/v1',
            'http://22.7.240.133:10070/v1',
        ])   
    else:
        base_url = args.api_base
    client = OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=600,
    )
    item = deepcopy(line_dict)
    if line_dict['initial_response'] is None:
        item['status'] = 'FAILED'
        img_path = line_dict['messages'][1]['content'][1]['image_url']['url']
        data_uri = image_to_data_uri(pathlib.Path(img_path))
        line_dict['messages'][1]['content'][1]['image_url']['url'] = data_uri
        completion = call_api_with_retry(args, client, line_dict['messages'])
        if not completion:
            item['status'] = 'API_CALL_FAILED'
            return item
        else:
            initial_response = completion
            code_content = extract_python_code(initial_response)
            if not code_content:
                item['status'] = 'NO_CODE'
            else:
                item['initial_response'] = code_content
                image_score = Image_Scoring(args, scoring_client, item)
                code_score = Code_Scoring(args, scoring_client, item)
                if image_score is None and code_score is None:
                    item['status'] = 'Scoring FAILED'
                elif code_score is None:
                    item['status'] = 'Code Scoring FAILED'
                    item['image_score'] = image_score
                elif image_score is None:
                    item['status'] = 'Image Scoring FAILED'
                    item['code_score'] = code_score
                else:
                    item['status'] = 'SUCCESS'
                    item['image_score'] = image_score
                    item['code_score'] = code_score
                save_result_safely(item, output_filepath, lock)
                return item
    else:
        print("正在处理半成功任务...")
        print(f"code_score: {item['code_score']}, image_score: {item['image_score']}")
        if item['code_score'] is None:
            image_score = Code_Scoring(args, scoring_client, item)
            item['code_score'] = image_score
        if item['image_score'] is None:
            code_score = Image_Scoring(args, scoring_client, item)
            item['image_score'] = code_score
        if item['code_score'] is None and item['image_score'] is None:
            item['status'] = 'Scoring FAILED'
        elif item['code_score'] is None:
            item['status'] = 'Code Scoring FAILED'
        elif item['image_score'] is None:
            item['status'] = 'Image Scoring FAILED'
        else:
            item['status'] = 'SUCCESS'
        save_result_safely(item, output_filepath, lock)
        return item
        
def get_model_params(name: str) -> dict:
    if name == "doubao-seed-1-6-vision-250815-nothinking":
        return {
            "model": "doubao-seed-1-6-vision-250815",
            "temperature": 0.01,
            "top_p": 0.001,
            "retry": 10,
            "max_tokens": 16384,
            "thinking": {"type": "disabled"},
        }
    elif name == "doubao-seed-1-6-vision-250815-thinking":
        return {
            "model": "doubao-seed-1-6-vision-250815",
            "temperature": 0.01,
            "top_p": 0.001,
            "retry": 10,
            "max_tokens": 16384,
        }
    elif name == "doubao-1-5-thinking-vision-pro-250428":
        return {
            "model": "doubao-1-5-thinking-vision-pro-250428",
            "temperature": 0.01,
            "top_p": 0.001,
            "retry": 10,
            "max_tokens": 16384,
        }
    elif name == "claude-opus-4-1-20250805-thinking":
        return {
            "model": "claude-opus-4-1-20250805",
            "temperature": 1,
            "retry": 10,
            "max_tokens": 20000,
            "dashscope_extend_params": {"provider": "b"},
            "thinking": {"type": "enabled", "budget_tokens": 16384},
        }
    elif name == "claude-opus-4-1-20250805-nothinking":
        return {
            "model": "claude-opus-4-1-20250805",
            "temperature": 0.01,
            "top_p": 0.001,
            "retry": 10,
            "max_tokens": 16384,
        }
    elif name == "gemini_2.5_pro":
        return {
            "model": "gemini-2.5-pro",
            "temperature": 0,
            "retry": 10,
            "max_tokens": 16384,
        }
    elif name == "GPT5_high":
        return {
            "model": "gpt-5-2025-08-07",
            "retry": 10,
            "temperature": 1,
            "max_tokens": 16384,
            "reasoning_effort": "high",
            "verbose": False,
        }
    elif name == "GPT5_minimal":
        return {
            "model": "gpt-5-2025-08-07",
            "retry": 10,
            "temperature": 1,
            "max_tokens": 16384,
            "reasoning_effort": "minimal",
            "verbose": False,
        }
    elif name == "gemini_2.5_pro_bucket":
        return {
            "model": "gemini-2.5-pro",
            "retry": 10,
            "temperature": 1,
            "max_tokens": 16384,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled", "budget_tokens": 128} # 注释这行的话默认为adaptive thinking
        }
    elif name == "gemini_2.5_flash_bucket":
        return {
            "model": "gemini-2.5-flash",
            "retry": 10,
            "temperature": 1,
            "max_tokens": 16384,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled", "budget_tokens": 128} # 注释这行的话默认为adaptive thinking
        }
    elif name == "gemini_2.5_flash":
        return {
            "model": "gemini-2.5-flash",
            "retry": 10,
            "temperature": 1,
            "max_tokens": 16384,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-plus":
        return {
            "model": "qwen3-vl-plus",
            "retry": 10,
            "max_tokens": 16384,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-plus-2025-09-23":
        return {
            "model": "qwen3-vl-plus-2025-09-23",
            "retry": 10,
            "max_tokens": 16384,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-plus-2025-09-23-nothinking":
        return {
            "model": "qwen3-vl-plus-2025-09-23",
            "retry": 10,
            "max_tokens": 16384,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "disabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "4b_instruct":
        return {
            "model": "fix_aime_qwen3vl_4b_sft_cand1_distill_1st_from_maxpp_general755k_sample1_s4000_2nd_from_maxpp_general755k_sample1",
            "retry": 10,
            "max_tokens": 16384, 
            "temperature": 0.7,
            "extra_body":{'greedy':False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "4b_thinking":
        return {
            "model": "3vl_4b_distill",
            "retry": 10,
            "max_tokens": 40960, 
            "temperature": 1.0,
            "extra_body":{'greedy':False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.95, 'presence_penalty': 0.0, 'out_seq_length': 40960},
        }
    elif name == "8b_instruct":
        return {
            "model": "fix_aime_qwen3vl_8b_sft_cand1_distill_from_maxpp_general755k_sample1",
            "retry": 10,
            "max_tokens": 16384,
            "temperature": 0.7,
            "extra_body":{'greedy':False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "8b_thinking":
        return {
            "model": "3vl_8b_distill",
            "retry": 10,
            "max_tokens": 40960,
            "temperature": 1.0,
            "extra_body":{'greedy':False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.95, 'presence_penalty': 0.0, 'out_seq_length': 40960},
        }
    elif name == "qwen3-vl-32b-instruct":
        return {
            "model": "qwen3vl-32b-32k_s2432_256k_s70_distill_s2000_rl_s140_distill_aime_s50",
            "retry": 10,
            "max_tokens": 16384,
            "temperature": 0.7,
            "extra_body":{'greedy':False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "qwen3-vl-32b-thinking":
        return {
            "model": "Qwen3-VL-32B-Thinking",
            "retry": 10,
            "max_tokens": 40960,
            "temperature": 1.0,
            "extra_body":{'greedy':False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.95, 'presence_penalty': 0.0, 'out_seq_length': 40960},
        }
    elif name == "Qwen3-VL-8B-Instruct-CaptionCode":
        return {
            "model": "Qwen3-VL-8B-Instruct-CaptionCode",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "Qwen3-VL-8B-Instruct-CaptionCode-Stage2":
        return {
            "model": "Qwen3-VL-8B-Instruct-CaptionCode-Stage2",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "Qwen3-VL-4B-Instruct-CaptionCode":
        return {
            "model": "Qwen3-VL-4B-Instruct-CaptionCode",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "Qwen3-VL-235BA22-Instruct-CaptionCode":
        return {
            "model": "qwen3vl-235A22-tongkun-iter_0004986",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "Qwen3-VL-32B-Instruct-CaptionCode-noS":
        return {
            "model": "Qwen3-VL-32B-Instruct-CaptionCode-noS",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }        
    elif name == "Qwen3-VL-4B-Instruct-Code":
        return {
            "model": "Qwen3-VL-4B-Instruct-Code",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "Qwen3-VL-8B-Instruct-Code":
        return {
            "model": "Qwen3-VL-8B-Instruct-Code",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "CodePercept-R1-60":
        return {
            "model": "CodePercept-R1-60",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "CodePercept-R1-300":
        return {
            "model": "CodePercept-R1-300",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "CodePercept-S1-Code-Grpo-30":
        return {
            "model": "CodePercept-S1-Code-Grpo-30",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        } 
    elif name == "CodePercept-S1-Code-Grpo-90":
        return {
            "model": "CodePercept-S1-Code-Grpo-90",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        } 
    elif name == "CodePercept-S1-Code-Grpo-120":
        return {
            "model": "CodePercept-S1-Code-Grpo-120",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "CodePercept-S1-Code-Grpo-150":
        return {
            "model": "CodePercept-S1-Code-Grpo-150",
            "retry": 20,
            "max_tokens": 32768,
            "temperature": 0.7,
            "extra_body":{"enable_thinking": False, 'repetition_penalty': 1, 'top_k': 20, 'top_p':0.8, 'presence_penalty': 1.5, 'out_seq_length': 40960},
        }
    elif name == "MiMo-VL-7B-RL":
        return {
            "model": "MiMo-VL-7B-RL",
            "retry": 20,
            "max_tokens": 32768,
            "extra_body":{"enable_thinking": True, 'out_seq_length': 40960},
        }
    elif name == "Keye-VL-1_5-8B":
        return {
            "model": "Keye-VL-1_5-8B",
            "retry": 20,
            "max_tokens": 32768,
            "extra_body":{"enable_thinking": True, 'out_seq_length': 40960},
        }
    elif name == "Ovis2.5-9B":
        return {
            "model": "Ovis2.5-9B",
            "retry": 20,
            "max_tokens": 32768,
            "extra_body":{"enable_thinking": True, 'out_seq_length': 40960},
        }
    elif name == "InternVL3_5-8B":
        return {
            "model": "InternVL3_5-8B",
            "retry": 20,
            "max_tokens": 32768,
            "extra_body":{"enable_thinking": True, 'out_seq_length': 40960},
        }
    elif name == "MiMo-VL-7B-RL":
        return {
            "model": "MiMo-VL-7B-RL",
            "retry": 20,
            "max_tokens": 32768,
            "extra_body":{"enable_thinking": True, 'out_seq_length': 40960},
        }
    elif name == "GLM-4.1V-9B-Base":
        return {
            "model": "GLM-4.1V-9B-Base",
            "retry": 20,
            "max_tokens": 32768,
        }
    elif name == "Intern-S1-mini":
        return {
            "model": "Intern-S1-mini",
            "retry": 20,
            "max_tokens": 32768,
        }
    elif name == "MiniCPM-V-4_5":
        return {
            "model": "MiniCPM-V-4_5",
            "retry": 20,
            "max_tokens": 32768,
        }
    elif name == "qwen3-vl-235b-a22b-thinking":
        return {
            "model": "qwen3-vl-235b-a22b-thinking",
            "retry": 20,
            "max_tokens": 32768,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-235b-a22b-instruct":
        return {
            "model": "qwen3-vl-235b-a22b-instruct",
            "retry": 20,
            "max_tokens": 32768,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "disabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-32b-instruct":
        return {
            "model": "qwen3-vl-32b-instruct",
            "retry": 20,
            "max_tokens": 32768,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "disabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-30b-a3b-instruct":
        return {
            "model": "qwen3-vl-30b-a3b-instruct",
            "retry": 20,
            "max_tokens": 32768,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "disabled"} # 注释这行的话默认为adaptive thinking
        }
    elif name == "qwen3-vl-30b-a3b-thinking":
        return {
            "model": "qwen3-vl-30b-a3b-thinking",
            "retry": 20,
            "max_tokens": 32768,
            "dashscope_extend_params": {"provider": "b"}, # b表示google，d表示yingmao，根据拥堵程度选择，最好走google
            "thinking": {"type": "enabled"} # 注释这行的话默认为adaptive thinking
        }  
    elif name == "qwen2.5-vl-72b-instruct":
        return {
            "model": "qwen2.5-vl-72b-instruct",
            "retry": 20,
            "max_tokens": 8192,
        }  
    else:
        raise ValueError(f"Unknown model name: {name}")


def try_catch_debug_retry(args):
    Keys = [args.api_key]
    BASE_DATA_PATH = args.BASE_DATA_PATH
    BASE_OUTPUT_PATH = args.BASE_OUTPUT_PATH

    # 定义输出文件和图片保存目录
    output_jsonl_file = os.path.join(BASE_OUTPUT_PATH, f"{args.out_name}.jsonl")
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    args.output_image_dir = os.path.join(BASE_OUTPUT_PATH, f"{args.out_name}")
    os.makedirs(args.output_image_dir, exist_ok=True)

    # 1. [新增] 获取已完成的任务，实现断点续传
    completed_uuids = get_completed_uuids(output_jsonl_file)
    print(f"找到 {len(completed_uuids)} 个已成功完成的任务，将在本次运行中跳过。")

    # 2. 准备所有未完成的API请求
    api_requests = create_api_request_jsonl(
        base_data_path=BASE_DATA_PATH,
        completed_uuids=completed_uuids, # 传入已完成列表
        model_name=args.name,
        json_name = args.jsonl,
        output_path = output_jsonl_file,
    )

    if not api_requests:
        print("没有需要处理的新任务，程序退出。")
    else:
        print(f"已准备 {len(api_requests)} 个新任务，开始处理...")
        tasks_with_keys = []
        for i, request in enumerate(api_requests):
            api_key = Keys[i%len(Keys)]
            tasks_with_keys.append((api_key, request))
        manager = Manager()
        lock = manager.Lock()
        num_proc = args.nproc
        if num_proc == 1:
            for tasks_with_key in tasks_with_keys:
                call_single(tasks_with_key, args, output_jsonl_file, lock)
        worker_func = partial(call_single, 
                                args=args,
                                output_filepath=output_jsonl_file,
                                lock=lock,
                            )

        # 4. 执行多进程任务
        all_results = []
        with Pool(num_proc) as pool:
            # 使用 imap_unordered 来获取结果，这样可以实时更新进度条
            with tqdm(total=len(tasks_with_keys), desc="Processing Tasks") as pbar:
                for result in pool.imap_unordered(worker_func, tasks_with_keys):
                    all_results.append(result)
                    pbar.update(1)

        print(f"\n--- 所有任务处理完毕，结果已增量保存到 {output_jsonl_file} ---")
        
        # 统计最终结果 (从内存中的结果统计，也可以重新读取文件统计)
        try:
            stats = {}
            if len(all_results) == 0:
                print("没有处理任何任务。")
            else:
                print(f"已处理 {len(all_results)} 个任务。")
                for row in all_results:
                    status = row.get('status', 'FAILED')
                    stats[status] = stats.get(status, 0) + 1

                print("\n--- 本次运行任务统计 ---")
                if not stats:
                    print("本次运行没有处理任何任务。")
                else:
                    for status, count in sorted(stats.items()):
                        print(f"{status}: {count}")
        except Exception as e:
            print(f"统计结果时发生错误：{e}")
        print("------------------")


def arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Math2Cde.")
    parser.add_argument("--BASE_DATA_PATH", type=str, default="/home/data2/sgtk/benchmark")
    parser.add_argument("--BASE_OUTPUT_PATH", type=str, default="/home/data2/sgtk/evaluate_benchmark")
    parser.add_argument("--image_scoring_model", type=str, default="gemini-2.5-pro", help="Path to the model.")
    parser.add_argument("--code_scoring_model", type=str, default="gpt-4o-2024-11-20", help="Path to the model.")
    parser.add_argument("--api_key", type=str, default="sk-xxx")
    parser.add_argument("--api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--temper", type=float, default=1, help="Path to the model.")
    parser.add_argument("--name", type=str, default='gemini-2.5-pro', help="Path to the model.")
    parser.add_argument("--out_name", type=str, default='gemini-2.5-pro', help="Path to the model.")
    parser.add_argument("--jsonl", type=str, default="STEM2Code1012", help="Path to the model.")
    parser.add_argument("--max_retries", type=int, default=20, help="Path to the model.")
    parser.add_argument("--mode", type=str, default='requests', help="Path to the model.")
    parser.add_argument("--nproc", type=int, default=64, help="Path to the model.")
    return parser.parse_args()

for i in range(50):
    args = arg_parser()
    args.out_name = args.name
    try_catch_debug_retry(args)

"""
1.进程数不能设置大了 否则图像运行不成功 导致分很低
2.代码里面以及集成了 "requests" "openai"两种调用模式
3.很多模型已经支持 但需要配置好api_key api_base
运行命令:
python evaluation.py \
    --BASE_DATA_PATH "xxx" \
    --BASE_OUTPUT_PATH "xxx" \
    --api_key "sk-xxx" \
    --api_base "xxx" \
    --name "gemini-2.5-pro" \
    --out_name "gemini-2.5-pro" \
    --jsonl "STEM2Code" \
    --nproc 32
"""
