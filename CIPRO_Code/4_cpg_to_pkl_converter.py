# 合并节点并初步生成PKL文件
#from asyncio.windows_events import NULL
import os
import pickle
import networkx as nx
from pyparsing import null_debug_action
from parse_dot import parse_dot  # 导入解析DOT文件的函数
from collections import defaultdict

def merge_nodes_and_edges(graph_data, c_code_lines):
    merged_nodes = {} 
    merged_edges = []

    for node in graph_data['nodes']:
        node_id = node['id']
        line_number = node['line_number']

        if line_number in merged_nodes:
            # 如果这个行号已经出现过，合并到已有的节点中
            existing_node = merged_nodes[line_number]
            existing_node['ids'].append(node_id)
        else:
            # 否则，创建一个新的节点
            merged_nodes[line_number] = {
                'ids': [node_id],
                'line_number': line_number,
                'degree_centrality': None,  # 初始化为 None，稍后将填充这些值
                'closeness_centrality': None,
                'katz_centrality': None,
                'code_content': None
            }

    if c_code_lines is not None:
        for line_number, node_data in merged_nodes.items():
            if 0 < line_number <= len(c_code_lines):
                node_data['code_content'] = c_code_lines[line_number - 1]

    for graph_type in ['DDG', 'CDG', 'CFG']:
        edges = graph_data[graph_type]['edges']
        G = nx.DiGraph()

        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']

            source_line = None
            target_line = None

            for line, node_data in merged_nodes.items():
                if source_id in node_data['ids']:
                    source_line = node_data['line_number']
                if target_id in node_data['ids']:
                    target_line = node_data['line_number']

            if source_line and target_line:
                merged_edges.append((graph_type, source_line, target_line))

                # 计算中心性指标
                G.add_edge(source_line, target_line)
        
        # 计算中心性指标
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        katz_centrality = nx.katz_centrality(G)

        # 将中心性指标添加到节点属性
        for line_number, node_data in merged_nodes.items():
            node_data['degree_centrality'] = degree_centrality.get(line_number, None)
            node_data['closeness_centrality'] = closeness_centrality.get(line_number, None)
            node_data['katz_centrality'] = katz_centrality.get(line_number, None)

    return merged_nodes, merged_edges

dot_directory = "/home/sec/guilingxz/dataset/FFMPeg+Qemu/FQ_normalized_CPG/FQ_normalized_dot/No_Vul"  # 包含DOT文件的目录路径
c_directory = "/home/sec/guilingxz/dataset/FFMPeg+Qemu/outputs_c_normalized/No_Vul"  # 包含C文件的目录路径
output_directory = "/home/sec/guilingxz/dataset/FFMPeg+Qemu/FQ_normalized_CPG/raw_pkl_FQ_normalized/No_Vul"

# 获取所有DOT文件列表
dot_files = [f for f in os.listdir(dot_directory) if f.endswith('.dot')]

for dot_file in dot_files:
    try:
        function_name = os.path.splitext(dot_file)[0]

        # 创建数据结构来存储节点和边信息
        graph_data = {
            'nodes': [],  # 存储所有节点信息
            'vulnerable_nodes': [],
            'vulnerable_line': None,
            'AST': {'edges': []},
            'CFG': {'edges': []},
            'DDG': {'edges': []},
            'CDG': {'edges': []}
        }

        # 解析DOT文件并进行节点合并
        dot_file_path = os.path.join(dot_directory, dot_file)
        with open(dot_file_path, 'r') as file:
            dot_data = file.read()
            parse_dot(dot_data, graph_data)

        # 找到同名的C文件并获取与DOT文件中的行号相关的代码行
        c_file_path = os.path.join(c_directory, f"{function_name}.c")
        if os.path.exists(c_file_path):
            with open(c_file_path, 'r') as c_file:
                c_code_lines = c_file.readlines()
        else:
            print(f"无法找到C文件：{c_file_path}")
            c_code_lines = None
        # 调试输出
        # print("Code Lines:")
        # print(c_code_lines)
        # 合并相同行号的节点和处理边的关系
        merged_nodes, merged_edges = merge_nodes_and_edges(graph_data, c_code_lines)

        # 创建一个数据结构，将有向边点信息、代码行内容和C文件名组织在一起
        edge_and_code_data = {
            'function_name': function_name,
            'merged_nodes': list(merged_nodes.values()),  # 将 merged_nodes 字典转换为列表
            'merged_edges': merged_edges,
            'code_lines': c_code_lines
        }

        # 打印数据以进行验证
        print("Edge and Code Data:")
        print(edge_and_code_data)

        # 保存数据到Pickle文件
        output_filename = os.path.splitext(dot_file)[0] + '.pkl'
        output_path = os.path.join(output_directory, output_filename)

        with open(output_path, 'wb') as output_file:
            pickle.dump(edge_and_code_data, output_file)

        print(f"处理完成：{dot_file}，数据已保存到 {output_path}")
    except:
        print("error! ",dot_file)
        pass