import subprocess
import os
import shutil
import pandas as pd
from tqdm import tqdm
# path-to-joern-parse
JOERNPATH="/home/se/joern-cli/"
root_dir = './data/'
# location of source code dirs
source_dir = "./data/c_code/dirs"

def parse_source_code_to_dot(file_path, f, out_dir_pdg='/parsed/dot/pdg/', out_dir_cpg='/parsed/dot/cpg/'):
    root_path = './data'
    try :
        os.makedirs(root_path+out_dir_pdg)
        os.makedirs(root_path+out_dir_cpg)
    except:
        pass
    out_dir_cpg=root_path + '/parsed/dot/cpg/'
    # parse source code into cpg
    print('parseing source code into cpg...')
    shell_str = "sh " + JOERNPATH + "./joern-parse " + file_path
    subprocess.call(shell_str, shell=True) 
    print('exporting cpg from cpg root...')
    # 导出cpg的dot文件到指定的文件夹中
    # 本处指定的是：out_dir_cpg + fname
    # 即：'./data/parsed/dot/cpg/'
    shell_export_cpg = "sh " + JOERNPATH + "joern-export " + "--repr cpg14 --out " + out_dir_cpg + f.split('.')[0] + os.sep
    subprocess.call(shell_export_cpg, shell=True)

def main_func(source_dir = "./data/c_code/dirs", out_dir_cpg="解析好的cpg的dot文件的存放路径"):
    # all source code files, each file include a .cpp file
    dirs = os.listdir(source_dir)
    for c_folder in tqdm(dirs):
    	# 待解析的文件夹路径
        file_path = source_dir + '/' + c_folder
        # 解析好的cpg的dot文件的存放路径
        cpg_path = out_dir_cpg + '/' + c_folder
        # 如果cpg_path已经存在且解析出来的.dot文件个数大于0，说明该文件夹之前已经解析过了，就不再解析，避免重复工作
        if os.path.exists(cpg_path) and len(os.listdir(cpg_path)) > 0:
            print(f'{file_path} file has been processed')
            continue
        print(f'starting to process {file_path}')
        # 得到待解析的.cpp文件的名称 "filename.cpp"
        f = os.listdir(file_path)[0]
        #print(f)
        parse_source_code_to_dot(file_path, f)

# 调用main_func函数批量解析c++文件
if __name__ == "__main__":
    main_func()
