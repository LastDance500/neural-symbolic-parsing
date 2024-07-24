#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：neural-symbolic-parsing 
@File ：concept2id.py
@Author ：xiao zhang
@Date ：2023/12/12 22:44 
'''

import os
import argparse

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--sbn_file', default=os.path.join(path,"data/seq2lps/de/train/gold.sbn"), type=str,
                        help="path of sbn file, one independent sbn should be in one line")
    parser.add_argument("-s", '--save_file', default=os.path.join(path,"data/seq2id/de/train/gold.sbn"), type=str,
                        help="path to save sbn file")
    parser.add_argument("-t", '--idx_file', default=os.path.join(path,"pre-processing/utils/all_languages"), type=str,
                        help="path of id code file")
    args = parser.parse_args()
    return args


def create_mapping(idx_path):
    idx_mapping = {}
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = f.readlines()

    for t in idx:
        idx_code = t.split(" ")[0].strip()
        synset = t.split(" ")[2].strip()

        idx_mapping[synset] = idx_code

    return idx_mapping


def replace_synset(sbn_path, save_path, n_mapping, v_mapping, a_mapping, t_mapping, r_mapping, o_mapping):
    # read sbn file
    with open(sbn_path, "r", encoding="utf-8") as f:
        sbn = f.readlines()

    text = []
    replaced_sbn = []
    for s in sbn:
        text.append(s.split("\t")[0].strip())
        s_split = s.split("\t")[1].strip().split(" ")
        for i in range(len(s_split)):
            item = s_split[i].strip()
            if n_mapping.get(item):
                idx = n_mapping[item]
            elif v_mapping.get(item):
                idx = v_mapping[item]
            elif a_mapping.get(item):
                idx = a_mapping[item]
            elif t_mapping.get(item):
                idx = t_mapping[item]
            elif r_mapping.get(item):
                idx = r_mapping[item]
            elif o_mapping.get(item):
                idx = o_mapping[item]
            else:
                idx = item
            s_split[i] = idx
        replaced_sbn.append(" ".join(s_split))
    with open(save_path, "w", encoding="utf-8") as w:
        for i in range(len(replaced_sbn)):
            w.write(text[i] + "\t" + replaced_sbn[i] + "\n")


if __name__ == '__main__':
    # create args
    args = create_arg_parser()

    input_path = args.sbn_file
    save_path = args.save_file
    idx_path = args.idx_file

    # read idx code file, there should be three files(noun, verb, (adj and adv) and Role)
    n_idx_path = os.path.join(idx_path, "n.tax") # noun
    v_idx_path = os.path.join(idx_path, "v.tax") # verb
    a_idx_path = os.path.join(idx_path, "a.tax") # adj and adv
    t_idx_path = os.path.join(idx_path, "t.tax") # thematic role
    r_idx_path = os.path.join(idx_path, "r.tax") # thematic role
    o_idx_path = os.path.join(idx_path, "o.tax") # thematic role

    # creating mapping from synset to idx
    n_mapping = create_mapping(n_idx_path)
    v_mapping = create_mapping(v_idx_path)
    a_mapping = create_mapping(a_idx_path)
    t_mapping = create_mapping(t_idx_path)
    r_mapping = create_mapping(r_idx_path)
    o_mapping = create_mapping(o_idx_path)

    # replace the synset with the idx code
    replace_synset(input_path, save_path, n_mapping, v_mapping, a_mapping, t_mapping, r_mapping, o_mapping)
