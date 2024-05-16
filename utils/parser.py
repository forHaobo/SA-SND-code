"""
@FileName：parser.py
@Description：用于配置模型参数
@Author：HaoBoli
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'TCM', type = str,
                        help = 'Dataset to use. (MaFengWo or CAMRa2011)')
    parser.add_argument('--emb_dim', default = 32, type = int,
                        help = 'embedding size for hidden layer')
    parser.add_argument('--lr', default = 0.001, type = float,
                        help = 'learning rate')
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'epoch number')
    parser.add_argument('--batch_size', default = 64, type = int,
                        help = 'batch size')
    parser.add_argument('--layer_num', default = 2, type = int,
                        help = 'number of GCN layers')
    parser.add_argument('--weight_name', default = 'out', type = str,
                        help = 'weight file name')
    parser.add_argument('--test_num', default = 1, type = int,
                        help = 'number of test epoch')
    parser.add_argument('--model', default = 'my', type = str,
                        help = 'Select Model light: LightGCN; my: my model')
    parser.add_argument('--numSym', default = 3, type = int,
                        help = 'number of symptoms')
    parser.add_argument('--st', default = 0.2, type = float,
                        help = 'Similarity Threshold')
    parser.add_argument('--reg', default = 0.7, type = float,
                        help = 'symptom Regularization')
    parser.add_argument('--isReg', default = True,
                        help = 'is symptom Regularization')

    args = parser.parse_args()
    return args