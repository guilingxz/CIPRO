import os
import pickle
import sent2vec


def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def no_smooth(code_list, centrality_lists):
    return code_list, centrality_lists


def smooth_code_list_with_adaptive_split(code_list, centrality_lists, k=7):
    smoothed_code_list = []
    smoothed_centrality_lists = [[] for _ in centrality_lists]

    for i in range(len(code_list) - 1):
        smoothed_code_list.append(code_list[i])

        if i < len(code_list) - 1:
            left_code = code_list[i]
            right_code = code_list[i + 1]

            left_length = len(left_code)
            right_length = len(right_code)

            num_left_tokens = [int(left_length * (k - n) / k)
                               for n in range(1, k)]
            num_right_tokens = [int(right_length * n / k) for n in range(1, k)]

            left_tokens = []
            right_tokens = []

            for n in range(1, k):
                # 在左侧和右侧分别取代码行进行拼接
                left_tokens_n = left_code[-num_left_tokens[n-1]:]
                right_tokens_n = right_code[:num_right_tokens[n-1]]

                # 处理左侧和右侧的拼接行单词
                if right_tokens_n and right_tokens_n[-1].isalpha():
                    j = num_right_tokens[n-1]
                    while j < right_length and right_code[j].isalpha():
                        right_tokens_n += right_code[j]
                        j += 1

                if left_tokens_n and left_tokens_n[0].isalpha():
                    j = -num_left_tokens[n-1] - 1
                    while j >= -left_length and left_code[j].isalpha():
                        left_tokens_n = left_code[j] + left_tokens_n
                        j -= 1

                # 添加到列表
                left_tokens.append(left_tokens_n)
                right_tokens.append(right_tokens_n)

            # 创建多个拼接后的代码行
            concatenated_code_list = [
                left_tokens[n-1] + right_tokens[n-1] for n in range(1, k)]
            smoothed_code_list.extend(concatenated_code_list)

            for j in range(len(centrality_lists)):
                left_centrality = centrality_lists[j][i]
                right_centrality = centrality_lists[j][i + 1]

                # 处理 left_centrality 和 right_centrality 为 None 的情况
                left_centrality = left_centrality if left_centrality is not None else 0
                right_centrality = right_centrality if right_centrality is not None else 0

                smoothed_centrality_lists[j].append(left_centrality)
                for n in range(1, k):
                    smoothed_centrality = (
                        (k - n) / k) * left_centrality + (n / k) * right_centrality
                    smoothed_centrality_lists[j].append(smoothed_centrality)

    if len(code_list) > 0:
        smoothed_code_list.append(code_list[-1])

        for j in range(len(centrality_lists)):
            if len(smoothed_centrality_lists[j]) <= i:
                smoothed_centrality_lists[j].append(None)
            else:
                smoothed_centrality_lists[j].append(centrality_lists[j][-1])

    return smoothed_code_list, smoothed_centrality_lists


def process_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as pkl_file:
        edge_and_code_data = pickle.load(pkl_file)

    # 从 edge_and_code_data 中获取 merged_nodes 列表
    merged_nodes = edge_and_code_data['merged_nodes']

    # 将 merged_nodes 按 line_number 排序
    sorted_nodes = sorted(merged_nodes, key=lambda node: node['line_number'])

    # 创建一个列表，用于存储代码内容
    #code_list = [node['code_content'] for node in sorted_nodes]
    code_list = [node['code_content'].strip() for node in sorted_nodes]

    # 创建三个中心性指标列表
    degree_centrality_list = [node['degree_centrality']
                              for node in sorted_nodes]
    closeness_centrality_list = [
        node['closeness_centrality'] for node in sorted_nodes]
    katz_centrality_list = [node['katz_centrality'] for node in sorted_nodes]
    print("="*100)
    print("degree_centrality_list:", degree_centrality_list)
    print("closeness_centrality_list:", closeness_centrality_list)
    print("closeness_centrality_list:", closeness_centrality_list)
    # 调用平滑函数
    smoothed_code_list, smoothed_centrality_lists = smooth_code_list_with_adaptive_split(
        code_list, [degree_centrality_list, closeness_centrality_list, katz_centrality_list])
    print("=============================================================")
    print("94 smoothed_code_list:", smoothed_code_list)
    print("len smoothed_code_list:", len(smoothed_code_list))
    print("=============================================================")
    print("95 smoothed_centrality_lists:", smoothed_centrality_lists)
    print("len smoothed_centrality_lists:", len(smoothed_centrality_lists[0]))
    # 执行新的功能
    degree_channel = []
    closeness_channel = []
    katz_channel = []

    for code, (degree_cen, closeness_cen, katz_cen) in zip(smoothed_code_list, zip(*smoothed_centrality_lists)):
        # 将每行代码向量化为行向量
        # 请确保你的 sentence_embedding 函数能够将代码向量化
        line_vec = sentence_embedding(code)

        if line_vec is not None and degree_cen is not None and closeness_cen is not None and katz_cen is not None:
            # 将行向量乘以中心性指标值并添加到对应的通道中
            degree_channel.append(degree_cen * line_vec)
            closeness_channel.append(closeness_cen * line_vec)
            katz_channel.append(katz_cen * line_vec)
    # print(degree_channel, closeness_channel, katz_channel)

    return (degree_channel, closeness_channel, katz_channel)


def main():
    # 指定存储 pkl 文件的目录路径
    pkl_directory = "/home/user1/projects/guilingxz/dataset/new_data_no_extend/raw_pkl/No_Vul"
    out_directory = "/home/user1/projects/guilingxz/dataset/new_data_no_extend/img_pkl_k7/No_Vul"
    trained_model_path = "/home/user1/projects/guilingxz/project/VulCNN/data/data_model.bin"
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
