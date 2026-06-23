import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 设置模型路径
model_name_or_path = "./models/bert-base-uncased"

# 加载预训练的BERT模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path)
def sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def smooth_code_list_with_adaptive_split(code_list, merged_edges, centrality_lists, node_content_and_centrality_list):
    smoothed_code_list = []
    smoothed_centrality_lists = [[] for _ in centrality_lists]
    ddg_nodes = set()

    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'CFG':
            ddg_nodes.add(source_line)
            ddg_nodes.add(target_line)

    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'CFG':
            left_code = code_list[source_line - 1]
            right_code = code_list[target_line - 1]
            adaptive_split = int((len(left_code) + len(right_code)) / 2)
            # left_tokens = left_code[-adaptive_split//2:]
            # right_tokens = right_code[:adaptive_split//2]
            concatenated_code = left_code + right_code

            # if smoothed_code_list:
            #     smoothed_code_list[-1] += code_list[source_line - 1]
            if smoothed_code_list:
                smoothed_code_list.insert(len(smoothed_code_list), code_list[source_line - 1])

            smoothed_code_list.append(concatenated_code)

            for k in range(len(centrality_lists)):
                left_centrality = centrality_lists[k][source_line - 1]
                right_centrality = centrality_lists[k][target_line - 1]
                if smoothed_centrality_lists[k]:
                    smoothed_centrality_lists[k].append(left_centrality)
                smoothed_centrality = (left_centrality + right_centrality) / 2 if left_centrality is not None and right_centrality is not None else 0
                smoothed_centrality_lists[k].append(smoothed_centrality)

    for i in range(len(code_list)):
        if i + 1 not in ddg_nodes:
            smoothed_code_list.insert(i, code_list[i])
            for k in range(len(centrality_lists)):
                smoothed_centrality_lists[k].append(centrality_lists[k][i] if centrality_lists[k][i] is not None else 0)
    print(50*"=")
    print("codelist:",code_list)
    print("smooth_list:",smoothed_code_list)
    print("Code list length:", len(code_list))
    print("Smoothed code list length:", len(smoothed_code_list))
    print("Smoothed degree centrality list length:", len(smoothed_centrality_lists[0]))
    print(50*"=")
    return smoothed_code_list, smoothed_centrality_lists

def process_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as pkl_file:
        edge_and_code_data = pickle.load(pkl_file)

    merged_nodes = edge_and_code_data['merged_nodes']
    merged_edges = list(set(edge_and_code_data['merged_edges']))
    sorted_nodes = sorted(merged_nodes, key=lambda node: node['line_number'])
    code_and_centrality_list = []

    for node in sorted_nodes:
        line_number = node['line_number']
        if len(code_and_centrality_list) + 1 != line_number:
            code_and_centrality_list.extend([(None, None, None, None)] * (line_number - len(code_and_centrality_list) - 1))
        code_and_centrality_list.append((node['code_content'], node['degree_centrality'], node['closeness_centrality'], node['katz_centrality']))

    code_list = [x[0] for x in code_and_centrality_list]
    degree_centrality_list = [x[1] for x in code_and_centrality_list]
    closeness_centrality_list = [x[2] for x in code_and_centrality_list]
    katz_centrality_list = [x[3] for x in code_and_centrality_list]
    print("Code list length:", len(code_list))
    print("Degree centrality list length:", len(degree_centrality_list))
    print("Closeness centrality list length:", len(closeness_centrality_list))
    print("Katz centrality list length:", len(katz_centrality_list))
    smoothed_code_list, smoothed_centrality_lists = smooth_code_list_with_adaptive_split(
        code_list, merged_edges, [degree_centrality_list, closeness_centrality_list, katz_centrality_list], code_and_centrality_list)
    print("=============================================================")
    print("Smoothed code list length:", len(smoothed_code_list))
    print("Smoothed degree centrality list length:", len(smoothed_centrality_lists[0]))
    print("Smoothed closeness centrality list length:", len(smoothed_centrality_lists[1]))
    print("Smoothed katz centrality list length:", len(smoothed_centrality_lists[2]))
    print("=============================================================")
    degree_channel, closeness_channel, katz_channel = [], [], []

    for code, (degree_cen, closeness_cen, katz_cen) in zip(smoothed_code_list, zip(*smoothed_centrality_lists)):
        if code is not None:
            line_vec = np.array(sentence_embedding(code))
            degree_channel.append(line_vec*degree_cen)
            closeness_channel.append(line_vec*closeness_cen)
            katz_channel.append(line_vec*katz_cen)

    return degree_channel, closeness_channel, katz_channel

def main():
    pkl_directory = "/hdd1/project/dataset/Reveal/Reveal_CPG/raw_pkl_Reveal/Vul"
    out_directory = "/hdd1/project/dataset/Reveal/Reveal_CPG/img_pkl_CFG_bert_gray_Reveal/Vul"
    os.makedirs(out_directory, exist_ok=True)

    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

    for i, pkl_file in enumerate(pkl_files):
        pkl_file_path = os.path.join(pkl_directory, pkl_file)
        print(f"Processing file: {pkl_file_path}")
        try:
            channels = process_pkl_file(pkl_file_path)
            if channels:
                degree_channel, closeness_channel, katz_channel = channels
                out_pkl = os.path.join(out_directory, pkl_file)
                data = [degree_channel, closeness_channel, katz_channel]
                with open(out_pkl, 'wb') as f:
                    pickle.dump(data, f)
                print(f"File saved to: {out_pkl}")
        except Exception as e:
            print(f"Error processing file {pkl_file_path}: {e}")

if __name__ == "__main__":
    main()
