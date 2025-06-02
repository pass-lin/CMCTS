'''
我们从Gakao-en数据集中采样了一部分数据。我们在Native-MCTS-PRM和CMCTS-PRM上多次运行推理程序，并收集了所有生成的状态。
可视化native和cmcts的state
需要注意的是，为了避免因为超参数的影响。我们在运行这个脚本的时候top-p设置为1，top-k设置为512
这是因为原来的设置topp和topk都不会开很低，这种情况下我觉得不能真正展示他们所采样的状态多样性的上线。
'''

import json
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE  # 使用 t-SNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # 用于三维绘图
from sklearn.decomposition import PCA
import  umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# 设置环境变量
os.environ["TRANSFORMERS_DISABLE_TF"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 加载模型和 tokenizer
model_path = "jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()

# 加载历史数据

with open("history_cmcts_only_understant_repeat.json", "r") as f:
    cmcts_historys_understant = json.load(f)
with open("history_native_repeat.json", "r") as f:
    native_historys = json.load(f)


with open("history_cmcts_repeat.json", "r") as f:
    cmcts_historys = json.load(f)

def collect(history):
    texts = []
    for h in history:
        if type(h)==str:
            texts.append(h)
            continue
        text = ""
        for question,answer in h:
            text += question + "\n" + answer + "\n"
            #把每一个state存储起来,后面做可视化
            texts.append(text)
    embeddings = model.encode(texts, task="text-matching")

    return embeddings
# 为每个 history 创建单独的可视化


pdf_path = "visualization.pdf"

# 创建一个包含四个子图的网格布局
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(20, 20))
    #总共16个，可视化我选择了10 13 3 7这四个，你自己看情况，总共16个
    for idx, i in enumerate([10, 13, 3, 7]):
        
        cmcts_sentence_understant = collect(cmcts_historys_understant[i]['level_list'])
        native_sentence = collect(native_historys[i]['level_list'])
        cmcts_sentence = collect(cmcts_historys[i]['level_list'])
        all_sentences = np.concatenate([cmcts_sentence_understant, native_sentence,cmcts_sentence], axis=0)
        
        # 使用 UMAP 进行二维降维
        reducer = umap.UMAP()
        embeddings_2d = reducer.fit_transform(all_sentences)
        
        # 在2x2的子图网格中绘制每个图表
        ax = plt.subplot(2, 2, idx + 1)
        
        # 绘制 Native 点（绿色）
        native_embeddings = embeddings_2d[len(cmcts_sentence_understant):len(cmcts_sentence_understant)+len(native_sentence)]
        ax.scatter(native_embeddings[:, 0], native_embeddings[:, 1], c='cyan', alpha=0.5, label='Nativ MCTS')
        
        # 绘制 CMCTS 点（蓝色）
        cmcts_embeddings = embeddings_2d[:len(cmcts_sentence_understant)]
        ax.scatter(cmcts_embeddings[:, 0], cmcts_embeddings[:, 1], c='blue', alpha=0.5, label='CMCTS Only Understant')
        
        cmcts_embeddings = embeddings_2d[len(cmcts_sentence_understant)+len(native_sentence):len(cmcts_sentence_understant)+len(native_sentence)+len(cmcts_sentence)]
        ax.scatter(cmcts_embeddings[:, 0], cmcts_embeddings[:, 1], c='red', alpha=0.5, label='CMCTS')
        
        # 添加标题和标签
        title = native_historys[i]["question"]["question"].replace("\\lvert","").replace("\\rvert","").replace("?","?\n").replace(".",".\n")
        
        if title[-1]=="\n":
            title = title[:-1]
        ax.set_title(title.replace("$\n","$"))
        ax.set_xlabel('umap Dimension 1')
        ax.set_ylabel('umap Dimension 2')
        ax.legend()
        ax.grid(True)
    
    # 调整子图之间的间距以避免重叠
    plt.tight_layout()
    
    # 将当前图表保存到PDF
    pdf.savefig()
    plt.show()
    plt.close()  # 关闭当前图表以释放内存

print(f"可视化图表已保存到 {pdf_path}")

