#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：neural-symbolic-parsing 
@File ：WuP_similarity.py
@Author ：xiao zhang
@Date ：2023/12/14 19:08 
'''

import math


def wup(tc1, tc2):
    # Compare the first character; if different, return 0 immediately
    if tc1[0] != tc2[0]:
        return 0

    # Check for equal length after removing the prefix
    if len(tc1) != len(tc2):
        raise AttributeError("Two tax codes have different lengths!")

    # If codes are identical, return 1.0
    if tc1 == tc2:
        return 1.0

    # Find the first occurrence of '0' to determine the depth of each code
    depth_tc1 = tc1.find('0') if '0' in tc1 else len(tc1)
    depth_tc2 = tc2.find('0') if '0' in tc2 else len(tc2)

    # Determine the depth of the common prefix
    common_prefix = next((i for i, (c1, c2) in enumerate(zip(tc1, tc2)) if c1 != c2), depth_tc1)

    # Calculate the Wu-Palmer similarity
    return 2 * common_prefix / (depth_tc1 + depth_tc2)


def wup_penalty(tc1, tc2):
    # the formulation is Sim=Sim_wup*PF(tc1, tc2)
    if tc1[0] != tc2[0]:
        return 0

    tax_depth = len(tc1)

    # Check for equal length after removing the prefix
    if len(tc1) != len(tc2):
        raise AttributeError("Two tax codes have different lengths!")

    # If codes are identical, return 1.0
    if tc1 == tc2:
        return 1.0

    # Find the first occurrence of '0' to determine the depth of each code
    depth_tc1 = tc1.find('0') if '0' in tc1 else len(tc1)
    depth_tc2 = tc2.find('0') if '0' in tc2 else len(tc2)

    # Determine the depth of the common prefix
    common_prefix = next((i for i, (c1, c2) in enumerate(zip(tc1, tc2)) if c1 != c2), depth_tc1)

    if common_prefix == depth_tc1 or common_prefix == depth_tc2:
        # return Wu-Palmer similarity
        return 2 * common_prefix / (depth_tc1 + depth_tc2)
    else:
        # penalty
        pf = math.exp(-1 * (depth_tc1- common_prefix + depth_tc2 - common_prefix + 1) / tax_depth)
        return 2 * common_prefix / (depth_tc1 + depth_tc2) * pf


if __name__ == '__main__':
    tax_code_1 = "n1222213124112000000000"
    tax_code_2 = "n1222212113423100000000"

    print(wup(tax_code_1, tax_code_2))
    print(wup_penalty(tax_code_1, tax_code_2))