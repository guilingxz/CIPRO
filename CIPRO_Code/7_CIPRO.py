import os
import argparse
import sys
import torch
from model import CNN_Classifier, TraditionalDataset, get_MCM_score, load_data
# from model_bert import CNN_Classifier, TraditionalDataset, get_MCM_score, load_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import plot_confusion_matrix


def parse_options():
    parser = argparse.ArgumentParser(description='VulCNN training.')
    parser.add_argument('-i', '--input', help='The dir path of train.pkl and test.pkl', type=str, required=True)
    args = parser.parse_args()
    return args

def get_kfold_dataframe(pathname = "./data/", item_num = 0):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")[item_num]
    eval_df = load_data(pathname + "val.pkl")[item_num]
    # test_df = eval_df.copy(deep=True)
    return train_df, eval_df

def main():
    """
    python VulCNN.py; /usr/bin/shutdown      # 用;拼接意味着前边的指令不管执行成功与否，都会执行shutdown命令
    python VulCNN.py && /usr/bin/shutdown    # 用&&拼接表示前边的命令执行成功后才会执行shutdown。请根据自己的需要选择
    """
    #args = parse_options()
    item_num = 0
    #hidden_size = 768
    hidden_size = 128
    #data_path = args.input
    # data_path = "./new_data_no_extend_pkl"
    # model_save_path = "./new_data_no_extend_pkl/model2/"
    # log_folder = "./logs_new_data_no_extend_pkl2"
    # data_path = "../../autodl-tmp/new_data_vulcnn_normalized_prased_pkl"
    # model_save_path = "../../autodl-tmp/new_data_vulcnn_normalized_prased_pkl/model/"
    # log_folder = "../../autodl-tmp/logs_new_data_vulcnn_normalized_prased_pkl"
    # data_path = "../../autodl-tmp/old_no_extend_normalized_pkl"
    # model_save_path = "../../autodl-tmp/old_no_extend_normalized_pkl/model/"
    # log_folder = "../../autodl-tmp/logs_old_no_extend_normalized_pkl"
    # autodl-tmp/pkl_FQ_CFG_extend_lc_20241115
    # autodl-tmp/pkl_Reveal_CFG_extend_lc_20241115
    # autodl-tmp/pkl_sard_CFG_extend_lc_20241115
    data_path = "../../autodl-tmp/pkl_sard_CFG_extend_lc_20241115"
    model_save_path = "../../autodl-tmp/pkl_sard_CFG_extend_lc_20241115/model/"
    log_folder = "../logs_pkl_sard_CFG_extend_lc_20241115_100"
    os.makedirs(model_save_path, exist_ok=True)
    # Create the logs folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_file_path = os.path.join(log_folder, 'output.log')

    # Redirect stdout to both console and file
    orig_stdout = sys.stdout
    with open(log_file_path, 'w') as f:
        sys.stdout = f

        for item_num in range(1):
            train_df, eval_df = get_kfold_dataframe(pathname=data_path, item_num=item_num)
            classifier = CNN_Classifier(result_save_path=data_path.replace("pkl", "results"), \
                                        item_num=item_num, epochs=100, hidden_size=hidden_size)
            classifier.preparation(
                X_train=train_df['data'],
                y_train=train_df['label'],
                X_valid=eval_df['data'],
                y_valid=eval_df['label'],
            )
            classifier.train(model_save_path + str(item_num))

    # Restore stdout to the original
    sys.stdout = orig_stdout


if __name__ == "__main__":
    main()