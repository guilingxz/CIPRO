import os
import glob
import pickle
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, KFold

def save_data(filename, data):
    print(f"开始保存数据至：{filename}")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    print(f"开始读取数据于：{filename}")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_dataframe(input_path, save_path):
    # Ensure the paths end with a slash
    input_path = input_path.rstrip("/") + "/"
    save_path = save_path.rstrip("/") + "/"
    
    # Create save path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    records = []
    for type_name in os.listdir(input_path):
        dir_path = os.path.join(input_path, type_name)
        label = 1 if type_name == "Vul" else 0
        filenames = glob.glob(dir_path + "/*.pkl")
        
        for file in filenames:
            data = load_data(file)
            records.append({
                "filename": os.path.basename(file).rstrip(".pkl"), 
                "length": len(data[0]), 
                "data": data, 
                "label": label
            })
    
    df = pd.DataFrame(records)
    save_data(os.path.join(save_path, "all_data.pkl"), df)

def gather_data(input_path, output_path):
    generate_dataframe(input_path, output_path)

def split_data(all_data_path, save_path, kfold_num):
    df = load_data(all_data_path)
    save_path = save_path.rstrip("/") + "/"
    
    seed = 0
    kf = KFold(n_splits=kfold_num, shuffle=True, random_state=seed)
    
    # Split data by label
    label_dfs = {label: df[df.label == label] for label in df.label.unique()}
    
    train_splits, val_splits, test_splits = {}, {}, {}
    
    for epoch in range(kfold_num):
        train_dict, val_dict, test_dict = {}, {}, {}
        
        for label, label_df in label_dfs.items():
            # Split data into train+val and test
            train_val_idx, test_idx = next(kf.split(label_df))
            train_val_data, test_data = label_df.iloc[train_val_idx], label_df.iloc[test_idx]
            
            # Split train+val into train and val
            train_idx, val_idx = next(kf.split(train_val_data))
            train_data, val_data = train_val_data.iloc[train_idx], train_val_data.iloc[val_idx]
            
            train_dict[label] = train_data
            val_dict[label] = val_data
            test_dict[label] = test_data
        
        train_splits[epoch] = pd.concat(train_dict.values(), ignore_index=True)
        val_splits[epoch] = pd.concat(val_dict.values(), ignore_index=True)
        test_splits[epoch] = pd.concat(test_dict.values(), ignore_index=True)
    
    # Save the splits as dictionaries
    save_data(os.path.join(save_path, "train.pkl"), train_splits)
    save_data(os.path.join(save_path, "val.pkl"), val_splits)
    save_data(os.path.join(save_path, "test.pkl"), test_splits)

def main():
    input_path = "../../autodl-tmp/img_pkl_CFG_gray_bert_Reveal"
    output_path = "../../autodl-tmp/pkl_CFG_gray_bert_Reveal/"
    kfold_num = 5
    
    gather_data(input_path, output_path)
    split_data(os.path.join(output_path, "all_data.pkl"), output_path, kfold_num)

if __name__ == "__main__":
    main()
