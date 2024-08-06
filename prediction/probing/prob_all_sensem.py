import numpy as np
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import seaborn as sns


def parse_embeddings(embed):
    embedding_dict = {}
    for e in embed:
        tmp = e.split(" ")
        embedding_dict[tmp[0]] = np.array(tmp[1:], dtype=np.float16)
    return embedding_dict


def get_word_embedding(text, target_word, tokenizer, model):
    # 检查CUDA是否可用，并获取CUDA设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将输入文本编码为ID并移动到CUDA设备
    input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(device)
    # 将目标词编码为ID并移动到CUDA设备
    target_word_ids = tokenizer(target_word, return_tensors='pt')['input_ids'][0][:-1].to(device)

    def find_subsequence_indices(sequence, subsequence):
        """找到子序列 subsequence 在序列 sequence 中的位置"""
        seq_len = len(sequence[0])
        sub_len = len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[0][i:i + sub_len], subsequence):
                return i, i + sub_len
        return None

    with torch.no_grad():
        # 将模型移动到CUDA设备
        model.to(device)

        # 生成模型输出
        generated_ids = model.generate(input_ids, output_attentions=True)
        outputs = model(input_ids=input_ids, decoder_input_ids=generated_ids)

        encoder_embedding = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
        decoder_embedding = outputs.logits  # [batch_size, seq_len, vocab_size]

    # 找到目标单词的编码器嵌入
    positions = find_subsequence_indices(input_ids, target_word_ids)
    if positions:
        start_pos, end_pos = positions
        # 平均目标词在编码器嵌入中的表示
        word_encoder_embedding = encoder_embedding[0, start_pos:end_pos, :].mean(dim=0)
    else:
        print(f"Warning: Target word '{target_word}' not found in the input text.")
        word_encoder_embedding = None

    # 获取解码器嵌入的平均值
    word_decoder_embedding = decoder_embedding.mean(dim=1).squeeze()

    return word_encoder_embedding, word_decoder_embedding


def visualize_embeddings(embeddings, concepts, layers, title):
    if not embeddings:
        print("No embeddings were found. Exiting.")
        return

    embeddings_array = np.array(embeddings)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    sns.set(style="whitegrid")  # 设置 seaborn 样式为白色网格
    plt.figure(figsize=(12, 12))

    markers = ['o', 's', 'X', 'D']  # 实心圆, 方块, 叉, 菱形
    layer_to_label = {0: 'Abstract', 1: 'General', 2: 'Specific', 3: 'Concrete'}
    color_palette = sns.color_palette("hsv", len(set(layers)))  # 使用 HSV 颜色空间

    texts = []
    for i, (embedding, concept, layer) in enumerate(zip(reduced_embeddings, concepts, layers)):
        # 为每个点绘制散点
        plt.scatter(embedding[0], embedding[1], color=color_palette[layer], marker=markers[layer % len(markers)],
                    label=layer_to_label[layer], s=30, edgecolor='black')
        # 添加文本并收集文本对象以后续调整
        text = plt.text(embedding[0], embedding[1], ' ' + concept, ha='right', va='bottom', fontsize=9, color='darkslategray')
        texts.append(text)

    # 调用 adjust_text 来自动调整文本位置，减少重叠
    adjust_text(texts, expand=(1.2, 1.4), arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # 处理图例，避免重复
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Specificity Levels", title_fontsize='13', fontsize='11', loc='best')

    plt.title(title, fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.grid(True)  # 开启网格
    plt.axis('equal')  # 设置轴比例相同

    plt.savefig(f"{title}.png")
    plt.show()  # 显示图形

def structure_maintenance(embeddings, indices):
    try:

        sum = 0

        layer3_embedding = []
        layer2_embedding = []
        layer1_embedding = []
        layer0_embedding = []

        # 根据索引将嵌入划分到不同层次
        for i in range(len(indices)):
            if indices[i] == 3:
                try:
                    layer3_embedding.append(embeddings[i])
                except Exception as e:
                    pass
            elif indices[i] == 2:
                try:
                    layer2_embedding.append(embeddings[i])
                except Exception as e:
                    pass
            elif indices[i] == 1:
                try:
                    layer1_embedding.append(embeddings[i])
                except Exception as e:
                    pass
            else:
                try:
                    layer0_embedding.append(embeddings[i])
                except Exception as e:
                    pass

        # 计算各层次嵌入的平均值
        l3_embedding = np.mean(layer3_embedding, axis=0) if layer3_embedding else None
        l2_embedding = np.mean(layer2_embedding, axis=0) if layer2_embedding else None
        l1_embedding = np.mean(layer1_embedding, axis=0) if layer1_embedding else None
        l0_embedding = np.mean(layer0_embedding, axis=0) if layer0_embedding else None

        # 计算余弦相似度函数
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            return dot_product / (norm_vec1 * norm_vec2)

        # 计算各层次嵌入之间的余弦相似度
        d30 = 1 - cosine_similarity(l3_embedding,
                                    l0_embedding) if l3_embedding is not None and l0_embedding is not None else None
        d31 = 1 - cosine_similarity(l3_embedding,
                                    l1_embedding) if l3_embedding is not None and l1_embedding is not None else None
        d20 = 1 - cosine_similarity(l2_embedding,
                                    l0_embedding) if l2_embedding is not None and l0_embedding is not None else None
        d21 = 1 - cosine_similarity(l2_embedding,
                                    l1_embedding) if l2_embedding is not None and l1_embedding is not None else None
        d10 = 1 - cosine_similarity(l1_embedding,
                                    l0_embedding) if l1_embedding is not None and l0_embedding is not None else None
        d32 = 1 - cosine_similarity(l3_embedding,
                                    l2_embedding) if l3_embedding is not None and l2_embedding is not None else None

        sum = 0
        if d20 > d10:
            sum += 1
        if d30 > d10:
            sum += 1
        if d31 > d21:
            sum += 1
        if d20 > d21:
            sum += 1
        if d31 > d32:
            sum += 1
        if d30 > d32:
            sum += 1

        return sum / 6

    except Exception as e:
        return 0


def calculate_true_ratio(truth_list):
    # 计算 True 的比例
    true_count = sum(truth_list)  # True 当作 1，False 当作 0，sum 会计算 True 的总数
    total_count = len(truth_list)  # 总数
    true_ratio = true_count / total_count if total_count else 0  # 避免除以零
    return true_ratio


import pandas as pd
from nltk.corpus import wordnet as wn


def has_at_least_four_layers(synset):
    """检查从给定同义词集开始是否有至少四层上位词。"""
    current_layer = 0
    while synset:
        hypernyms = synset.hypernyms()
        if not hypernyms:
            break
        synset = hypernyms[0]
        current_layer += 1
        if current_layer == 4:
            return True
    return False


if __name__ == '__main__':

    with open("asdjhjkzxh.txt", "w", encoding="utf-8") as w:

        data = pd.read_csv("../../data/seq2lps/en/test/challenge_extended.csv")

        try:
            with open("../../ares_embedding/sensembert_EN_supervised.txt", "r", encoding="utf-8") as f:
                embeddings = f.readlines()[1:]
            embedding_dict = parse_embeddings(embeddings)
        except Exception as e:
            pass

        results = []
        results_decoder = []
        for i in range(len(data)):
            text = data.iloc[i]['text']
            concept_name = data.iloc[i]['target_concept']


            target_word = data.iloc[i]['target_word']

            # 直接获取指定的同义词集
            current_synset = wn.synset(concept_name)

            # 检查是否有足够的层级
            if not has_at_least_four_layers(current_synset):
                continue  # 没有足够的层级，跳过这一项

            # 提取四层概念
            concepts = []
            layers = []

            # 添加当前同义词集及其sister terms
            concepts.append(current_synset.name())
            layers.append(3)
            count = 1  # 包括了当前同义词集

            # 添加sister terms
            for hyper in current_synset.hypernyms():
                for hypo in hyper.hyponyms():
                    if hypo != current_synset and count < 10:  # 限制每层最多十个synset
                        concepts.append(hypo.name())
                        layers.append(3)
                        count += 1
                    if count == 10:
                        break
                if count == 10:
                    break

            # 处理上位词
            hypernyms = current_synset.hypernyms()
            count = 0
            current_synset = hypernyms[0]
            concepts.append(current_synset.name())
            layers.append(2)
            for hyper in current_synset.hypernyms():
                for hypo in hyper.hyponyms():
                    if hypo != current_synset and count < 10:  # 限制每层最多十个synset
                        concepts.append(hypo.name())
                        layers.append(2)
                        count += 1
                    if count == 9:
                        break
                if count == 9:
                    break

            # 继续处理上位词
            hypernyms = current_synset.hypernyms()
            count = 0
            current_synset = hypernyms[0]
            concepts.append(current_synset.name())
            layers.append(1)
            for hyper in current_synset.hypernyms():
                for hypo in hyper.hyponyms():
                    if hypo != current_synset and count < 10:  # 限制每层最多十个synset
                        concepts.append(hypo.name())
                        layers.append(1)
                        count += 1
                    if count == 9:
                        break
                if count == 9:
                    break

            # 最后一层
            hypernyms = current_synset.hypernyms()
            current_synset = hypernyms[0]
            concepts.append(current_synset.name())
            layers.append(0)

            all_layers = layers

            try:
            # 加载 Sensembert 嵌
                sensembert_embeddings = []
                for concept in concepts:
                    sense_key = wn.synset(concept).lemmas()[0].key()
                    if sense_key in embedding_dict:
                        try:
                            sensembert_embeddings.append(embedding_dict[sense_key])
                        except Exception as e:
                            pass
                    # else:
                        # sensembert_embeddings.append(np.random.rand(2048))
                        # print(f"Warning: Embedding for sense key '{sense_key}' not found.")

                # visualize_embeddings(sensembert_embeddings, all_concepts, all_layers, 'PCA of Sensembert Embeddings')
                try:
                    r = structure_maintenance(sensembert_embeddings, all_layers)
                    results.append(r)
                    print(f"{r} + {concept_name}")
                except Exception as e:
                    pass
            except Exception as e:
                pass

            # # 模型 seq2tax 和 seq2lps 的嵌入
            # folder = "../../model_saves/run6/byt5/lps"
            # model_name = "PCA of seq2lps Embeddings"
            #
            # tokenizer = AutoTokenizer.from_pretrained(folder, max_length=512, output_attentions=True)
            # model = T5ForConditionalGeneration.from_pretrained(folder).to("cuda:0")
            #
            # model_embeddings_encoder = []
            # model_embeddings_decoder = []
            #
            # if target_word in text:
            #
            #     for concept in concepts:
            #         t = concept.split('.')[0].replace('_', ' ')
            #
            #         new_text = text.replace(target_word, t)
            #         embedding_encoder, embedding_decoder = get_word_embedding(new_text, t, tokenizer, model)
            #
            #         model_embeddings_encoder.append(embedding_encoder.to("cpu").numpy())
            #
            # r = structure_maintenance(model_embeddings_encoder, all_layers)
            # results.append(r)
            # w.write(str(r) + "\n")
            # w.flush()
            # print(str(r))

        print(calculate_true_ratio(results))

