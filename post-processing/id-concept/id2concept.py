#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：neural-symbolic-parsing 
@File ：tax2concept.py
@Author ：xiao zhang
@Date ：2023/12/15 19:30 
'''

import os
import argparse

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--sbn_file', default=os.path.join(path,"experiments/results/run4/byt5/id//standard/standard.sbn"), type=str,
                        help="path of sbn file, one independent sbn should be in one line")
    parser.add_argument("-s", '--save_file', default=os.path.join(path,"experiments/post_results/run4/byt5/id//standard/standard.sbn.post"), type=str,
                        help="path to save sbn file")
    parser.add_argument("-t", '--tax_file', default=os.path.join(path,"pre-processing/utils/wide"), type=str,
                        help="path of tax code file")
    parser.add_argument("-d", '--distance', default=False, type=bool,
                        help="using post-edit or not")
    args = parser.parse_args()
    return args


def create_mapping(tax_path):
    tax_mapping = {}
    with open(tax_path, "r", encoding="utf-8") as f:
        tax = f.readlines()

    for t in tax:
        tax_code = t.split(" ")[0].strip()
        synset = t.split(" ")[2].strip()

        tax_mapping[tax_code] = synset

    if tax_mapping.get("100001740"):
        tax_mapping["100001740"] = "entity.n.01"

    return tax_mapping


def replace_synset(sbn_path, save_path, n_mapping, v_mapping, a_mapping, t_mapping, r_mapping, o_mapping):
    # read sbn file
    with open(sbn_path, "r", encoding="utf-8") as f:
        sbn = f.readlines()
    replaced_sbn = []
    for s in sbn:
        s_split = s.strip().split(" ")
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
            w.write(replaced_sbn[i] + "\n")


if __name__ == '__main__':
    # create args
    args = create_arg_parser()

    input_path = args.sbn_file
    save_path = args.save_file
    tax_path = args.tax_file

    # read tax code file, there should be three files(noun, verb, (adj and adv) and Role)
    n_tax_path = os.path.join(tax_path, "n.tax") # noun
    v_tax_path = os.path.join(tax_path, "v.tax") # verb
    a_tax_path = os.path.join(tax_path, "a.tax") # adj and adv
    t_tax_path = os.path.join(tax_path, "t.tax") # thematic role
    r_tax_path = os.path.join(tax_path, "r.tax") # discourse role
    o_tax_path = os.path.join(tax_path, "o.tax") # discourse role

    # creating mapping from synset to tax
    n_mapping = create_mapping(n_tax_path)
    v_mapping = create_mapping(v_tax_path)
    a_mapping = create_mapping(a_tax_path)
    t_mapping = create_mapping(t_tax_path)
    r_mapping = create_mapping(r_tax_path)
    o_mapping = create_mapping(o_tax_path)

    # replace the synset with the tax code
    replace_synset(input_path, save_path, n_mapping, v_mapping, a_mapping, t_mapping, r_mapping, o_mapping)
