#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：neural-symbolic-parsing 
@File ：concept2tax.py
@Author ：xiao zhang
@Date ：2023/12/12 22:44 
'''

import os
import argparse

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--sbn_file', default=os.path.join(path,"data/seq2lps/de/dev/standard.sbn"), type=str,
                        help="path of sbn file, one independent sbn should be in one line")
    parser.add_argument("-s", '--save_file', default=os.path.join(path,"data/seq2tax/de/dev/standard.sbn"), type=str,
                        help="path to save sbn file")
    parser.add_argument("-t", '--tax_file', default=os.path.join(path,"pre-processing/utils/all_languages"), type=str,
                        help="path of tax code file")
    parser.add_argument("-p", '--prune', default=False, type=bool,
                        help="prune or not prune the tax code")
    args = parser.parse_args()
    return args


def prune_tax_code(s):
    if s[-1] == "+" or s[-1] == "-" or s[-1] == "|":
        tail = s[-1]
        s = s[:-1]
    else:
        tail = ""

    last_index = None
    for i, char in enumerate(s):
        if char.isdigit():
            if char != '0':
                last_index = i
        else:
            last_index = i

    if last_index is None:
        return s + tail

    return s[:last_index + 1] + tail


def create_mapping(tax_path, prune):
    tax_mapping = {}
    with open(tax_path, "r", encoding="utf-8") as f:
        tax = f.readlines()

    for t in tax:
        if not prune:
            tax_code = t.split(" ")[1].strip()
        else:
            tax_code = prune_tax_code(t.split(" ")[1].strip())
        synset = t.split(" ")[2].strip()
        tax_mapping[synset] = tax_code
    return tax_mapping


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
                tax = n_mapping[item]
            elif v_mapping.get(item):
                tax = v_mapping[item]
            elif a_mapping.get(item):
                tax = a_mapping[item]
            elif t_mapping.get(item):
                tax = t_mapping[item]
            elif r_mapping.get(item):
                tax = r_mapping[item]
            elif o_mapping.get(item):
                tax = o_mapping[item]
            else:
                tax = item

            s_split[i] = tax
        replaced_sbn.append(" ".join(s_split))

    with open(save_path, "w", encoding="utf-8") as w:
        for i in range(len(replaced_sbn)):
            w.write(text[i] + "\t" + replaced_sbn[i] + "\n")


if __name__ == '__main__':
    # create args
    args = create_arg_parser()

    input_path = args.sbn_file
    save_path = args.save_file
    tax_path = args.tax_file
    prune = args.prune

    # read tax code file, there should be three files(noun, verb, (adj and adv) and Role)
    n_tax_path = os.path.join(tax_path, "n.tax") # noun
    v_tax_path = os.path.join(tax_path, "v.tax") # verb
    a_tax_path = os.path.join(tax_path, "a.tax") # adj and adv
    t_tax_path = os.path.join(tax_path, "t.tax") # thematic role
    r_tax_path = os.path.join(tax_path, "r.tax") # discourse role
    o_tax_path = os.path.join(tax_path, "o.tax") # operator

    # creating mapping from synset to tax
    n_mapping = create_mapping(n_tax_path, prune)
    v_mapping = create_mapping(v_tax_path, prune)
    a_mapping = create_mapping(a_tax_path, prune)
    t_mapping = create_mapping(t_tax_path, prune)
    r_mapping = create_mapping(r_tax_path, prune)
    o_mapping = create_mapping(o_tax_path, prune)

    # replace the synset with the tax code
    replace_synset(input_path, save_path, n_mapping, v_mapping, a_mapping, t_mapping, r_mapping, o_mapping)

