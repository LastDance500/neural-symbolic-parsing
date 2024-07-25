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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from adjustText import adjust_text


def visualize_embeddings(embeddings, concepts, layers, title, average_cal):
    if not embeddings:
        print("No embeddings were found. Exiting.")
        return

    embeddings_array = np.array(embeddings)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 12))

    markers = ['o', 's', 'X', 'D']  # 实心圆, 方块, 叉, 菱形
    layer_to_label = {3: 'synonyms 0', 2: 'hypernyms 1', 1: 'hypernyms 2', 0: 'hypernyms 3'}
    color_palette = sns.color_palette("hsv", len(layer_to_label))

    # 对应的颜色映射
    color_mapping = {'7': 3, '6': 2, '5': 1, '4': 0, '3': 3, '2': 2, '1': 1, '0': 0}

    # 分组存储 embeddings 并计算平均
    group_embeddings = {str(i): [] for i in range(8)}
    for embedding, cal in zip(embeddings, average_cal):
        group_embeddings[str(cal)].append(embedding)

    average_embeddings = {cal: np.mean(group, axis=0) if group else None for cal, group in group_embeddings.items()}
    average_reduced_embeddings = {cal: pca.transform([emb])[0] if emb is not None else None for cal, emb in average_embeddings.items()}

    legend_labels = set()  # 用于确保图例不重复
    texts = []

    for i, (embedding, concept, layer) in enumerate(zip(reduced_embeddings, concepts, layers)):
        label = layer_to_label.get(layer, 'Unknown')  # 确保 layer 是正确的键类型
        marker = markers[layer % len(markers)]
        color = color_palette[layer]
        plt.scatter(embedding[0], embedding[1], color=color, marker=marker, s=30, edgecolor='slategrey', alpha=0.4, label=label if label not in legend_labels else "")
        if label not in legend_labels:
            legend_labels.add(label)
        text = plt.text(embedding[0], embedding[1], ' ' + concept, ha='right', va='bottom', fontsize=8, color='slategrey')
        texts.append(text)

    # 绘制平均 embeddings
    for cal, embedding in average_reduced_embeddings.items():
        if embedding is not None:
            layer_index = color_mapping[cal]
            marker = markers[layer_index % len(markers)]
            plt.scatter(embedding[0], embedding[1], color=color_palette[layer_index], marker=marker, s=100, edgecolor='black')
            text = plt.text(embedding[0], embedding[1], f'Avg. {layer_to_label[layer_index]}', ha='right', va='bottom', fontsize=11, color=color_palette[layer_index])
            texts.append(text)

    # 调整文本以减少重叠
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # 创建图例
    plt.legend(title="Specificity Levels", title_fontsize='13', fontsize='11', loc='best')

    plt.xlabel('Principal Component 1', fontsize=10)
    plt.ylabel('Principal Component 2', fontsize=10)
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"{title}.png", dpi=500)
    plt.show()


def structure_maintenance(embeddings, indices):
    layer3_embedding = []
    layer2_embedding = []
    layer1_embedding = []
    layer0_embedding = []

    # 根据索引将嵌入划分到不同层次
    for i in range(len(indices)):
        if indices[i] == 3:
            layer3_embedding.append(embeddings[i])
        elif indices[i] == 2:
            layer2_embedding.append(embeddings[i])
        elif indices[i] == 1:
            layer1_embedding.append(embeddings[i])
        else:
            layer0_embedding.append(embeddings[i])

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
    d30 = 1 - cosine_similarity(l3_embedding, l0_embedding) if l3_embedding is not None and l0_embedding is not None else None
    d31 = 1 - cosine_similarity(l3_embedding, l1_embedding) if l3_embedding is not None and l1_embedding is not None else None
    d20 = 1 - cosine_similarity(l2_embedding, l0_embedding) if l2_embedding is not None and l0_embedding is not None else None
    d21 = 1 - cosine_similarity(l2_embedding, l1_embedding) if l2_embedding is not None and l1_embedding is not None else None
    d10 = 1 - cosine_similarity(l1_embedding, l0_embedding) if l1_embedding is not None and l0_embedding is not None else None

    # 判断距离关系
    comparisons = {
        'd30 > d20': d30 > d20 if d30 is not None and d20 is not None else None,
        'd30 > d10': d30 > d10 if d30 is not None and d10 is not None else None,
        'd20 > d10': d20 > d10 if d20 is not None and d10 is not None else None,
        'd31 > d21': d31 > d21 if d31 is not None and d21 is not None else None,
    }

    print(comparisons)

    return comparisons


if __name__ == '__main__':

    noun = True
    verb = False
    adj = False
    # noun
    if noun:
        # concept1 = ['coat.n.03', 'beard.n.04', 'body_hair.n.01', 'cowlick.n.01', 'down.n.05', 'eyebrow.n.01', 'eyelash.n.01', 'facial_hair.n.01', 'forelock.n.02', 'guard_hair.n.01',
        #             'hair.n.01', 'epicranium.n.01', 'exoskeleton.n.01', 'exuviae.n.01', 'feather.n.01', 'headful.n.02', 'hide.n.02', 'protective_covering.n.02', 'skin.n.01',
        #             'body_covering.n.01', 'bark.n.01', 'blanket.n.02', 'crust.n.02', 'envelope.n.04', 'hood.n.02', 'indumentum.n.01', 'indusium.n.01', 'integument.n.01', 'perianth.n.01',
        #             'covering.n.01']

        # concept1 = ['grand_piano.n.01', 'mechanical_piano.n.01', 'upright.n.02',
        #             'piano.n.01', 'accordion.n.01', 'celesta.n.01', 'clavichord.n.01', 'clavier.n.02', 'organ.n.05', 'synthesizer.n.02', 'bones.n.01', 'chime.n.01', 'cymbal.n.01',
        #             'keyboard_instrument.n.01', 'barrel_organ.n.01', 'bass.n.07', 'calliope.n.02', 'electronic_instrument.n.01', "jew's_harp.n.01", 'music_box.n.01', 'percussion_instrument.n.01', 'stringed_instrument.n.01', 'wind_instrument.n.01',
        #             'musical_instrument.n.01']

        concept1 = ['drive.n.10', 'acoustic_device.n.01', 'adapter.n.02', 'afterburner.n.01', 'agglomerator.n.01', 'airfoil.n.01', 'alarm.n.02', 'appliance.n.01', 'applicator.n.01',
                    'device.n.01', 'ceramic.n.01', 'connection.n.03', 'container.n.01', 'conveyance.n.03', 'equipment.n.01', 'furnishing.n.02', 'hardware.n.02', 'implement.n.01', 'means.n.02',
                    'instrumentality.n.03', 'americana.n.01', 'anachronism.n.02', 'antiquity.n.03', 'article.n.02', 'block.n.01', 'building_material.n.01', 'button.n.07', 'commodity.n.01', 'cone.n.01',
                    'artifact.n.01']

        concept2 = ['almond.n.02', 'cherry.n.03', 'drupelet.n.01', 'elderberry.n.02', 'jujube.n.02', 'olive.n.04', 'peach.n.03', 'plum.n.02', 'beechnut.n.01', 'brazil_nut.n.02',
                    'drupe.n.01', 'accessory_fruit.n.01', 'achene.n.01', 'acorn.n.01', 'aggregate_fruit.n.01', 'berry.n.02', 'buckthorn_berry.n.01', 'buffalo_nut.n.01', 'chokecherry.n.01', 'cubeb.n.01',
                    'fruit.n.01', 'agamete.n.01', 'anther.n.01', 'antheridium.n.01', 'ascus.n.01', 'basidium.n.01', 'cone.n.03', 'endosperm.n.01', 'flower.n.02', 'fructification.n.02',
                    'reproductive_structure.n.01']

        # concept2 = ['drive.n.10', 'acoustic_device.n.01', 'adapter.n.02', 'afterburner.n.01', 'agglomerator.n.01', 'airfoil.n.01', 'alarm.n.02', 'appliance.n.01', 'applicator.n.01', 'aspergill.n.01',
        #             'device.n.01', 'ceramic.n.01', 'connection.n.03', 'container.n.01', 'conveyance.n.03', 'equipment.n.01', 'furnishing.n.02', 'hardware.n.02', 'implement.n.01', 'means.n.02',
        #             'instrumentality.n.03', 'americana.n.01', 'anachronism.n.02', 'antiquity.n.03', 'article.n.02', 'block.n.01', 'building_material.n.01', 'button.n.07', 'commodity.n.01', 'cone.n.01',
        #             'artifact.n.01']

    elif verb:
        pass
    else:
        # adj & adv
        concept1 = ["good.a.01", "satisfactory.a.01", "bad.a.01", "sad.a.03"]

        concept2 = ["angry.a.01", "mad.a.01", "livid.a.03", "unangry.a.01"]

        concept3 = ["badly.r.02", "ill.r.01", "healthily.r.01"]

        # all_concepts = concept1 + concept2 + concept3
    all_concepts = concept1 + concept2

    if noun:
        layers_concept1 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        layers_concept2 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    elif verb:
        layers_concept1 = [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0]
        layers_concept2 = [2, 2, 2, 2, 1, 1, 1, 1, 0]
        layers_concept3 = [2, 2, 2, 2, 1, 1, 1, 1, 0]
    else:
        layers_concept1 = [1, 1, 0, 0]
        layers_concept2 = [0, 0, 0, 1]
        layers_concept3 = [1, 1, 1, 1, 0]

    # all_layers = layers_concept1 + layers_concept2 + layers_concept3
    all_layers = layers_concept1 + layers_concept2
    average_cal = layers_concept1 + [i+4 for i in layers_concept2]

    try:
    # 加载 Sensembert 嵌入
        with open("../../ares_embedding/sensembert_EN_supervised.txt", "r", encoding="utf-8") as f:
            embeddings = f.readlines()[1:]
        embedding_dict = parse_embeddings(embeddings)

        sensembert_embeddings = []
        for concept in all_concepts:
            sense_key = wn.synset(concept).lemmas()[0].key()
            if sense_key in embedding_dict:
                sensembert_embeddings.append(embedding_dict[sense_key])
            else:
                print(f"Warning: Embedding for sense key '{sense_key}' not found.")

        print(1)
        visualize_embeddings(sensembert_embeddings, all_concepts, all_layers, 'PCA of Sensembert Embeddings', average_cal)

        structure_maintenance(sensembert_embeddings[:len(concept1)], all_layers[:len(concept1)])
        structure_maintenance(sensembert_embeddings[len(concept1): len(concept1) + len(concept2)],
                              all_layers[len(concept1): len(concept1) + len(concept2)])
        structure_maintenance(sensembert_embeddings[len(concept1) + len(concept2): len(concept1) + len(concept2) + len(concept3)],
                              all_layers[len(concept1) + len(concept2): len(concept1) + len(concept2) + len(concept3)])
    except Exception as e:
        print(e)

    # 模型 seq2tax 和 seq2lps 的嵌入
    for folder, model_name in [("../../model_saves/run7/byt5/tax/wide", "PCA of TAX-byt5 Embeddings"),
                               ("../../model_saves/run6/byt5/lps", "PCA of LPS-byt5 Embeddings"),
                               ("../../model_saves/run6/byt5/id", "PCA of WID-byt5 Embeddings")]:
        tokenizer = AutoTokenizer.from_pretrained(folder, max_length=512, output_attentions=True)
        model = T5ForConditionalGeneration.from_pretrained(folder).to("cuda:0")

        model_embeddings_encoder = []
        model_embeddings_decoder = []

        # example sentences
        if noun:
            for concept in tqdm(all_concepts, desc=f"Processing concepts for {folder}"):
                target_word = concept.split('.')[0].replace('_', ' ')
                if concept in concept2:
                    text1 = f"Tita prepared a mole with chocolate, {target_word}, and sesame."
                    embedding_encoder1, embedding_decoder1 = get_word_embedding(text1, target_word, tokenizer, model)

                    text2 = f"Emily enjoyed a {target_word} milkshake."
                    embedding_encoder2, embedding_decoder2 = get_word_embedding(text2, target_word, tokenizer, model)

                    text3 = f"The baker's specialty was {target_word} croissant."
                    embedding_encoder3, embedding_decoder3 = get_word_embedding(text3, target_word, tokenizer, model)

                    embedding_encoder = (embedding_encoder1 + embedding_encoder2 + embedding_encoder3) / 3
                    embedding_decoder = (embedding_decoder1 + embedding_decoder2 + embedding_decoder3) / 3

                elif concept in concept1:
                    text1 = f"The software engineer put the floppy into the {target_word}."
                    embedding_encoder1, embedding_decoder1 = get_word_embedding(text1, target_word, tokenizer, model)

                    text2 = f"He backed up the data onto an external {target_word}."
                    embedding_encoder2, embedding_decoder2 = get_word_embedding(text2, target_word, tokenizer, model)

                    text3 = f"She saved her work to the {target_word} for safekeeping."
                    embedding_encoder3, embedding_decoder3 = get_word_embedding(text3, target_word, tokenizer, model)

                    embedding_encoder = (embedding_encoder1 + embedding_encoder2 + embedding_encoder3) / 3
                    embedding_decoder = (embedding_decoder1 + embedding_decoder2 + embedding_decoder3) / 3

                else:
                    text1 = f"The soldier was shot in the {target_word}."
                    embedding_encoder1, embedding_decoder1 = get_word_embedding(text1, target_word, tokenizer, model)

                    text2 = f"He injured his {target_word} while jogging."
                    embedding_encoder2, embedding_decoder2 = get_word_embedding(text2, target_word, tokenizer, model)

                    text3 = f"She felt a sharp pain in her {target_word}."
                    embedding_encoder3, embedding_decoder3 = get_word_embedding(text3, target_word, tokenizer, model)

                    embedding_encoder = (embedding_encoder1 + embedding_encoder2 + embedding_encoder3) / 3
                    embedding_decoder = (embedding_decoder1 + embedding_decoder2 + embedding_decoder3) / 3

                if embedding_encoder is not None:
                    model_embeddings_encoder.append(embedding_encoder.to("cpu").numpy())
                    model_embeddings_decoder.append(embedding_decoder.to("cpu").numpy())

        elif verb:
            for concept in tqdm(all_concepts, desc=f"Processing concepts for {folder}"):
                target_word = concept.split('.')[0].replace('_', '')
                if concept in concept1:
                    text = f"I {target_word} to the students about the importance of teamwork."
                elif concept in concept2:
                    text = f"I {target_word} across the park every morning."
                else:
                    text = f"At the wildlife reserve, the animals {target_word} every evening."

                embedding_encoder, embedding_decoder = get_word_embedding(text, target_word, tokenizer, model)
                if embedding_encoder is not None:
                    model_embeddings_encoder.append(embedding_encoder.numpy())
                    model_embeddings_decoder.append(embedding_decoder.numpy())

        else:
            for concept in tqdm(all_concepts, desc=f"Processing concepts for {folder}"):
                target_word = concept.split('.')[0].replace('_', '')
                if concept in concept1:
                    text = f"The audience was {target_word} with this film."
                elif concept in concept2:
                    text = f"He was {target_word} after hearing the news."
                else:
                    text = f"He lived {target_word} throughout the year."

                embedding_encoder, embedding_decoder = get_word_embedding(text, target_word, tokenizer, model)
                if embedding_encoder is not None:
                    model_embeddings_encoder.append(embedding_encoder.numpy())
                    model_embeddings_decoder.append(embedding_decoder.numpy())

        visualize_embeddings(model_embeddings_encoder, all_concepts, all_layers, model_name+" encoder", average_cal)

        print(f"{model_name} encoder:")
        structure_maintenance(model_embeddings_encoder[:len(concept1)], all_layers[:len(concept1)])
        structure_maintenance(model_embeddings_encoder[len(concept1):],
                              all_layers[len(concept1):])
        # structure_maintenance(
        #     model_embeddings_encoder[len(concept1) + len(concept2): len(concept1) + len(concept2) + len(concept3)],
        #     all_layers[len(concept1) + len(concept2): len(concept1) + len(concept2) + len(concept3)])

        visualize_embeddings(model_embeddings_decoder, all_concepts, all_layers, model_name+" decoder", average_cal)

        print(f"{model_name} decoder:")
        structure_maintenance(model_embeddings_decoder[:len(concept1)], all_layers[:len(concept1)])
        structure_maintenance(model_embeddings_decoder[len(concept1):],
                              all_layers[len(concept1):])
        # structure_maintenance(
        #     model_embeddings_decoder[len(concept1) + len(concept2): len(concept1) + len(concept2) + len(concept3)],
        #     all_layers[len(concept1) + len(concept2): len(concept1) + len(concept2) + len(concept3)])

