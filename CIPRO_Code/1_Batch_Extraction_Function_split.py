import os
import re
import hashlib
from tqdm import tqdm  # 引入tqdm库

def extract_functions(source_code):
    functions = []
    in_function = False
    function_lines = []
    bracket_count = 0

    for line in source_code.splitlines():
        if not in_function:
            match = re.match(r"(\w+\s*\**\s+\w+\s*\([^)]*\))", line)
            if match:
                in_function = True
                function_lines = [match.group(1)]
                bracket_count = 0

        if in_function:
            function_lines.append(line)
            for char in line:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        in_function = False
                        function_code = "\n".join(function_lines).strip()
                        # 如果函数定义只有一行且包含分号，并且没有其他函数定义后面，则跳过
                        if len(function_lines) == 1 and ';' in function_lines[0] and not re.search(r"(\w+\s*\**\s+\w+\s*\([^)]*\))", source_code[source_code.index(function_code) + len(function_code):]):
                            break
                        functions.append(function_code)
                        break

    return functions

if __name__ == '__main__':
    input_folder = r'F:\dataset\SARD\raw_source'  # 输入文件夹路径
    output_folder = r'F:\dataset\SARD\raw_no_extend_function_slice'  # 输出文件夹路径

    # 获取输入文件夹中的所有源码文件，包括.c和.cpp文件
    source_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".c", ".cpp"))]

    # 使用tqdm创建进度条
    for source_file in tqdm(source_files, desc="Processing files"):
        with open(source_file, 'r') as f:
            code = f.read()
            functions = extract_functions(code)

            for idx, function_code in enumerate(functions):
                # 计算函数代码的哈希值，并将其转换为十六进制字符串
                function_code_hash = hashlib.md5(function_code.encode()).hexdigest()
                # 构建唯一的目标输出文件名
                function_filename = f"{idx}_{function_code_hash}.c"
                target_file = os.path.join(output_folder, function_filename)

                try:
                    # 保存函数代码到目标输出路径，去掉第一行重复的函数定义
                    with open(target_file, 'w') as f:
                        f.write("\n".join(function_code.splitlines()[1:]) + "\n")
                except OSError as e:
                    print("写入文件失败:", function_filename)
                    print("错误信息:", str(e))
