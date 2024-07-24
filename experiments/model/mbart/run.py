#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DRG_parsing 
@File ：run.py
@Author ：xiao zhang
@Date ：2022/11/14 12:27
'''

import argparse
import os
import sys
sys.path.append(".")

from model import get_dataloader, Generator

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", required=False, type=str, default="en",
                        help="language in [en, nl, de ,it]")
    parser.add_argument("-ip", "--if_pre", required=False, action='store_true',
                        default=False,
                        help="if pre-train or not")
    parser.add_argument("-pt", "--pretrain", required=False, type=str,
                        default=os.path.join(path, "data/seq2tax/en/train/gold_silver.sbn"),
                        help="pre-train(fine-tuning) sets")
    parser.add_argument("-t", "--train", required=False, type=str,
                        default=os.path.join(path, "data/seq2tax/en/train/gold.sbn"),
                        help="train sets")
    parser.add_argument("-d", "--dev", required=False, type=str,
                        default=os.path.join(path, "data/seq2lps/en/train/gold.sbn"),
                        help="dev sets")
    parser.add_argument("-e", "--test", required=False, type=str,
                        default=os.path.join(path, "data/seq2lps/en/test/standard.sbn"),
                        help="standard test sets")
    parser.add_argument("-mp", "--model_path", required=False, type=str,
                        default="",
                        help="path to load the trained model")
    parser.add_argument("-c", "--challenge", nargs='*', required=False, type=str,
                        default=os.path.join(path, "data/seq2lps/en/test/challenge.sbn"),
                        help="challenge sets")
    parser.add_argument("-s", "--save", required=False, type=str,
                        default=os.path.join(path, "model/byT5/tax_p_result"),
                        help="path to save the result")
    parser.add_argument("-epoch", "--epoch", required=False, type=int,
                        default=16)
    parser.add_argument("-lr", "--learning_rate", required=False, type=float,
                        default=1e-04)
    parser.add_argument("-ms", "--model_save", required=False, type=str,
                        default=os.path.join(path, "trained_model/seq2lps/gold"))
    args = parser.parse_args()
    return args


def ensure_directory(path):
    """ Ensure directory exists. """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    args = create_arg_parser()

    # Ensure save directories exist
    ensure_directory(args.save)
    ensure_directory(os.path.join(args.save, "standard"))
    ensure_directory(os.path.join(args.save, "challenge"))
    ensure_directory(args.model_save)

    # Train process
    lang = args.lang

    # Train loader
    train_dataloader = get_dataloader(args.train)

    # Test loader
    test_dataloader = get_dataloader(args.test)
    dev_dataloader = get_dataloader(args.dev)

    # Hyperparameters
    epoch = args.epoch
    lr = args.learning_rate

    # load the model
    if os.path.exists(args.model_path):
        model_path = args.model_path
    else:
        model_path = ""

    # Train
    bart_classifier = Generator(lang, load_path=model_path)
    if args.if_pre:  # if pretrain or not
        train_dataloader_pre = get_dataloader(args.pretrain)
        bart_classifier.train(train_dataloader_pre, dev_dataloader, lr=lr, epoch_number=3)
    bart_classifier.train(train_dataloader, dev_dataloader, lr=lr, epoch_number=epoch, save_path=args.model_save)

    # Standard test
    trained_bart_classifier = Generator(lang, load_path=args.model_save)
    trained_bart_classifier.evaluate(test_dataloader, os.path.join(args.save, "standard/standard.sbn"))

    # Challenge test
    for i, cha_path in enumerate(args.challenge):
        try:
            cha_dataloader = get_dataloader(cha_path)
            trained_bart_classifier.evaluate(cha_dataloader, os.path.join(args.save, f"challenge/challenge{i}.sbn"))
        except Exception as e:
            print(e)

    # bart_classifier.model.save_pretrained(args.model_save)


if __name__ == '__main__':
    main()
