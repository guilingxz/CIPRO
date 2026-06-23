# coding=utf-8
import os
import re
import argparse
from clean_gadget import clean_gadget

def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def normalize_directory(dir_path):
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".c"):  # Process only .c files
                filepath = os.path.join(root, filename)
                print(filepath)
                pro_one_file(filepath)

def pro_one_file(filepath):
    with open(filepath, "r") as file:
        code = file.read()

    code = re.sub(r'(?<!:)\/\/.*|\/\*(\s|.)*?\*\/', "", code, flags=re.DOTALL)

    with open(filepath, "w") as file:
        file.write(code.strip())

    with open(filepath, "r") as file:
        org_code = file.readlines()
        nor_code = clean_gadget(org_code)
    
    with open(filepath, "w") as file:
        file.writelines(nor_code)

def main():
    #args = parse_options()
    #normalize_directory(args.input)
    input_path = r'F:\dataset\SARD\no_extend_normalized_function_slice\good'
    normalize_directory(input_path)

if __name__ == '__main__':
    main()
