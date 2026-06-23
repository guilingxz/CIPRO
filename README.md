```markdown
# CIPRO

Official implementation of our paper: **CIPRO: A Pixel-Row Oversampling-Based Approach for Code Image Enhancement in Vulnerability Detection**

## Abstract

**Context:** In recent years, the rapid development of deep learning technologies has opened new avenues for vulnerability detection. Many existing methods convert source code into images for detection but often overlook the quality of the generated images, resulting in the loss of critical vulnerability features.

**Objectives:** This study aims to address the challenges of vulnerability feature loss and image quality issues in vulnerability detection. Specifically, we investigate the effectiveness of a novel pixel-row oversampling method based on code line concatenation for generating enhanced code images.

**Methods:** We propose a new system, CIPRO, which leverages pixel-row oversampling to integrate structural and semantic information from code. CIPRO generates interpolated code lines, creating continuous and feature-rich images for improved vulnerability detection. We validate this method through theoretical analysis and empirical studies on three datasets: Devign, Reveal, and SARD.

**Results:** Our experiments show that CIPRO achieves significant performance improvements, with F1-score increases of 5.36%, 19.62%, and 6.46% on Devign, Reveal, and SARD datasets, respectively, compared to existing methods like VulCNN. CIPRO also demonstrates high time efficiency and strong visualization capabilities, meeting practical large-scale requirements.

**Conclusion:** CIPRO's pixel-row oversampling method addresses the shortcomings of traditional code image generation methods, providing a scalable and effective solution for vulnerability detection.

**Keywords:** security, deep learning, program analysis, program representation

## Key Contributions

- **Identification of Image Quality Issues in Code Visualization:** For the first time, we highlight the issue of image quality in code visualization and propose a novel vulnerability detection framework, CIPRO. The framework employs a pixel-row oversampling method based on code control flow, effectively generating code feature images with enhanced pixel continuity between rows, significantly improving the recognition of vulnerability features.

- **Proposed Pixel-Row Oversampling Algorithm:** We introduce a pixel-row oversampling algorithm based on code line concatenation to address discontinuities in numerical values when directly converting code into images. By leveraging the node relationships in control flow graphs, new code lines are interpolated to balance local variations in the image and emphasize critical features. This method enhances CNNs' ability to extract key features, thereby improving classification performance.

- **Comprehensive Experimental Evaluation:** We conducted comparative experiments on three datasets, demonstrating that CIPRO outperforms five other vulnerability detection methods in both performance and computational efficiency.

## Dataset and Resources Download

Due to GitHub's file size restrictions, additional resources are hosted on **Quark Drive**:

| Resource Type | Download Link | Extraction Code |
|---------------|---------------|-----------------|
| Full Dataset (v1.2) | [Quark Drive](https://pan.quark.cn/s/89eeec6f8703) | `w3mF` |
| Sent2Vec Pre-trained Model | Same as above | `w3mF` |
| Joern Components | Same as above | `w3mF` |

**Notes:**
- Total size: ~11.6 GB (compressed)
- Please ensure stable internet connection when downloading.

## Scripts Description

| Script Name | Description |
|-------------|-------------|
| `1_Batch_Extraction_Function_split.py` | Batch extracts function-level code snippets from source files for analysis |
| `2_Linux_parse_source_code_to_dot.py` | Parses source code with Joern to generate code property graphs (CPGs) in DOT format (Linux environment) |
| `3_c_code_normalizer.py` | Normalizes C source code to a unified format to reduce noise |
| `4_cpg_to_pkl_converter.py` | Converts Code Property Graph (CPG) from DOT format to PKL format for efficient processing |
| `5_pixel_row_oversampler.py` | Implements the pixel-row oversampling technique for data augmentation to balance vulnerability samples |
| `6_kfold_dataset_splitter.py` | Splits the dataset into K-fold cross-validation sets for robust model evaluation |
| `7_CIPRO.py` | Main entry script for model training and evaluation |
| `clean_gadget.py` | Preprocesses and cleans gadget code snippets for further analysis |
| `model.py` | Defines the neural network architecture for vulnerability detection |
| `parse_dot.py` | Parses and processes DOT graph files for CPG extraction |
| `process_bert_ddg.py` | Processes Data Dependence Graphs (DDG) using BERT-based representation learning |
| `process_pkl_file_k_pro.py` | Processes PKL files with K-pro settings for graph-based feature extraction |

## Usage

Run the scripts sequentially to process the source code and train the vulnerability detection model:

```bash
python 1_Batch_Extraction_Function_split.py
python 2_Linux_parse_source_code_to_dot.py
python 3_c_code_normalizer.py
python 4_cpg_to_pkl_converter.py
python 5_pixel_row_oversampler.py
python 6_kfold_dataset_splitter.py
python 7_CIPRO.py
```

Adjust file paths and parameters inside each script as needed.

## Requirements

```
# Core dependencies
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0

# Deep learning (CUDA 10.2)
torch==1.12.1
torchvision==0.13.1
torchaudio==0.12.1

# Utilities
tqdm>=4.62.0
prettytable>=3.0.0

# Graph processing
networkx>=2.6.0

# Note: The following dependencies require special installation:
# - lap: conda install -c conda-forge lap
# - sent2vec: Needs manual compilation (https://github.com/facebookresearch/sent2vec)
```

## Reference

This project is based on the method described in the paper:

**CIPRO: A Pixel-Row Oversampling-Based Approach for Code Image Enhancement in Vulnerability Detection**

*Information and Software Technology* (Elsevier), 2026.
```
