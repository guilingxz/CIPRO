import os
import pickle
import sent2vec

def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    #print("emb[0]:",emb[0])
    return emb[0]
def no_smooth(code_list, centrality_lists):
    return code_list, centrality_lists

def smooth_code_list_with_adaptive_split(code_list, merged_edges, centrality_lists, node_content_and_centrality_list):
    # 初始化一个新的列表用于存储平滑后的代码
    print("Code list length:", len(code_list))
    print("centrality list length:", len(centrality_lists[0]))
    smoothed_code_list = []
    print("22 merged_edges:",merged_edges)
    # 初始化中心性指标列表，用于存储拼接行的中心性指标
    smoothed_centrality_lists = [[] for _ in centrality_lists]

    # 保存 DDG 边连接的节点
    ddg_nodes = set()

    # 遍历 DDG 边，记录连接的节点
    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'CFG':
            ddg_nodes.add(source_line)
            ddg_nodes.add(target_line)
    print("34 ddg_nodes:",ddg_nodes)

    # 遍历 DDG 边，按顺序拼接节点
    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'CFG' and source_line!=target_line:
            # 获取左侧和右侧代码行
            left_code = code_list[source_line - 1]
            right_code = code_list[target_line - 1]
            # 计算左侧和右侧的代码行长度
            left_length = len(left_code)
            right_length = len(right_code)
            # 计算自适应的拆分数量（示例中简化为平均值）
            adaptive_split = int((left_length + right_length) / 2)

            # # 在左侧和右侧分别取代码行进行拼接
            # left_tokens = left_code[-adaptive_split//2:]
            # right_tokens = right_code[:adaptive_split//2]

            # 拼接左侧和右侧代码行
            #concatenated_code = left_tokens + right_tokens
            concatenated_code = left_code + right_code

            # # 添加前一行的内容到拼接后的列表中
            # if smoothed_code_list:
            #     smoothed_code_list[-1] += code_list[source_line - 1]

            # 插入前一行的内容到拼接后的列表中的指定位置
            smoothed_code_list.insert(len(smoothed_code_list), code_list[source_line - 1])

            # 添加拼接后的代码行到平滑后的列表中
            smoothed_code_list.append(concatenated_code)

            # 将中心性指标添加到相应的列表中
            for k in range(len(centrality_lists)):
                left_centrality = centrality_lists[k][source_line - 1]
                right_centrality = centrality_lists[k][target_line - 1]

                # 计算平滑中心性
                if left_centrality is not None and right_centrality is not None:
                    smoothed_centrality = (left_centrality + right_centrality) / 2
                else:
                    smoothed_centrality = 0
                # 添加中心性值到列表中
                smoothed_centrality_lists[k].append(smoothed_centrality)

    # 将没有 DDG 边的节点添加到平滑后的列表中
    for i in range(len(code_list)):
        if i + 1 not in ddg_nodes:
            # 添加源代码行到平滑后的列表中 
            smoothed_code_list.insert(i, code_list[i])
            
            # 将中心性指标添加到相应的列表中
            for k in range(len(centrality_lists)):
                if centrality_lists[k][i] is not None:
                    smoothed_centrality_lists[k].append(centrality_lists[k][i])
                else:
                    smoothed_centrality_lists[k].append(0)

    print(50*"=")
    print("codelist:",code_list)
    print("smooth_list:",smoothed_code_list)
    print("Code list length:", len(code_list))
    print("Smoothed code list length:", len(smoothed_code_list))
    print("Smoothed degree centrality list length:", len(smoothed_centrality_lists[0]))
    print(50*"=")
    #print("cleaned_smoothed_code_list:",cleaned_smoothed_code_list)
    #return cleaned_smoothed_code_list, cleaned_smoothed_centrality_lists
    return smoothed_code_list, smoothed_centrality_lists


def process_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as pkl_file:
        edge_and_code_data = pickle.load(pkl_file)

    # 从 edge_and_code_data 中获取 merged_nodes 列表
    merged_nodes = edge_and_code_data['merged_nodes']
    merged_edges = edge_and_code_data['merged_edges']
    # 将 merged_nodes 按 line_number 排序
    sorted_nodes = sorted(merged_nodes, key=lambda node: node['line_number'])

    # 创建一个列表，用于存储源代码行和中心性指标值
    code_and_centrality_list = []

    # 遍历所有节点
    for node in sorted_nodes:
        line_number = node['line_number']

        # 检查节点行号是否连续
        if len(code_and_centrality_list) + 1 != line_number:
            # 如果节点行号不连续，则在列表中添加空的元素
            code_and_centrality_list.extend([(None, None, None, None)] * (line_number - len(code_and_centrality_list) - 1))

        # 将代码行和中心性指标值添加到列表中
        code_and_centrality_list.append((node['code_content'], node['degree_centrality'], node['closeness_centrality'], node['katz_centrality']))

    # 将空值替换为 None
    code_and_centrality_list = [(None if x[0] is None else x[0], None if x[1] is None else x[1], None if x[2] is None else x[2], None if x[3] is None else x[3]) for x in code_and_centrality_list]
    # 从 edge_and_code_data 中获取 merged_edges 列表，并进行去重
    merged_edges = list(set(edge_and_code_data['merged_edges']))
    # 将代码行和中心性指标值分开存储
    code_list = [x[0] for x in code_and_centrality_list]
    degree_centrality_list = [x[1] for x in code_and_centrality_list]
    closeness_centrality_list = [x[2] for x in code_and_centrality_list]
    katz_centrality_list = [x[3] for x in code_and_centrality_list]

    print("Code list length:", len(code_list))
    print("Degree centrality list length:", len(degree_centrality_list))
    print("Closeness centrality list length:", len(closeness_centrality_list))
    print("Katz centrality list length:", len(katz_centrality_list))

    # 调用平滑函数
    smoothed_code_list, smoothed_centrality_lists = smooth_code_list_with_adaptive_split(
        code_list, merged_edges, [degree_centrality_list, closeness_centrality_list, katz_centrality_list], code_and_centrality_list)

    print("=============================================================")
    print("Smoothed code list length:", len(smoothed_code_list))
    print("Smoothed degree centrality list length:", len(smoothed_centrality_lists[0]))
    print("Smoothed closeness centrality list length:", len(smoothed_centrality_lists[1]))
    print("Smoothed katz centrality list length:", len(smoothed_centrality_lists[2]))
    print("=============================================================")

    # 执行新的功能
    degree_channel = []
    closeness_channel = []
    katz_channel = []

    # for code, (degree_cen, closeness_cen, katz_cen) in zip(smoothed_code_list, zip(*smoothed_centrality_lists)):
    #     # 将每行代码向量化为行向量
    #     if code!=None:
    #         line_vec = sentence_embedding(code)
    #     else:
    #         line_vec==None
    #     if line_vec is not None and degree_cen is not None and closeness_cen is not None and katz_cen is not None:
    #         # 将行向量乘以中心性指标值并添加到对应的通道中
    #         # print("Type of degree_cen:", type(degree_cen))
    #         # print("Type of line_vec:", type(line_vec))
    #         degree_channel.append(line_vec)
    #         closeness_channel.append(line_vec)
    #         katz_channel.append(line_vec)

    for code in smoothed_code_list:
        # 将每行代码向量化为行向量
        if code is not None:
            line_vec = sentence_embedding(code)
        else:
            line_vec = None

        # 如果行向量存在，将其添加到所有通道中
        if line_vec is not None:
            degree_channel.append(line_vec)
            closeness_channel.append(line_vec)
            katz_channel.append(line_vec)

    # print(degree_channel)
    # print("Channel (sent2vec) size:", len(degree_channel), "x", len(degree_channel[0]) if degree_channel else 0)
    return (degree_channel, closeness_channel, katz_channel)

def main():
    # 指定存储 pkl 文件的目录路径
    # pkl_directory = "/hdd1/project/VulMCI/fun1_test"
    # out_directory = "/hdd1/project/VulMCI/fun1_test/img_pkl"
    # trained_model_path = "/hdd1/project/VulMCI/sent2vec/Reveal_model_128.bin"

    # pkl_directory = "/hdd1/project/dataset/Reveal/Reveal_CPG/raw_pkl_Reveal/No_Vul"
    # out_directory = "/hdd1/project/dataset/Reveal/Reveal_CPG/img_pkl_Reveal_CFG_new_insert/No_Vul"
    # trained_model_path = "/hdd1/project/VulMCI/sent2vec/Reveal_model_128.bin"

    pkl_directory = "/hdd1/project/VulMCI/fun_test_norm/raw_pkl"
    out_directory = "/hdd1/project/VulMCI/fun_test_norm/img_pkl"
    # trained_model_path = "/hdd1/project/VulMCI/sent2vec/Reveal_model_128.bin"
    trained_model_path = "/hdd1/project/VulMCI/data_model.bin"

    os.makedirs(out_directory, exist_ok=True)
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()  # type: ignore
    sent2vec_model.load_model(trained_model_path)
    # 获取目录中所有的 pkl 文件
    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

    for _, pkl_file in enumerate(pkl_files):
        pkl_file_path = os.path.join(pkl_directory, pkl_file)
        print("正在处理：", pkl_file_path)
        try:
            channels = process_pkl_file(pkl_file_path)
            if channels == None:
                return None
            else:
                (degree_channel, closeness_channel, katz_channel) = channels
                out_pkl = os.path.join(out_directory, pkl_file)
                data = [degree_channel, closeness_channel, katz_channel]
                with open(out_pkl, 'wb') as f:
                    pickle.dump(data, f)
                    # print("\r[{}/{}]文件已保存至：{}".format(_+1,len(pkl_files),out_pkl),end='')
                    print("文件已保存至：", out_pkl)
        except:
            print("")
            print("error:", pkl_file_path, end="\n")
            pass

if __name__ == "__main__":
    main()