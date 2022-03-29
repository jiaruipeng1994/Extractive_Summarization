#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Sat Jan 29 18:17:11 2022
# @Author : JRP - Ruipeng Jia

##################################################################
## Common
import string
printable = set(string.printable)

##################################################################
## Summary: Predictions & References
predictions = ['a b c d e', 'aa bb cc dd ee', '1 2 3 4 5', '11 22 33 44 55', 'hello world , hello jrp']
references = ['a f g h i', 'aa bb ff gg hh', '1 2 3 6 7', '11 22 33 44 66', 'hello world , hello jrp']

##################################################################
## sents_words_tokenize
from nltk import sent_tokenize, word_tokenize

def sents_words_tokenize(document):
    sents = [sent.strip() for sent in sent_tokenize(document)]
    sents_words = [word_tokenize(sent) for sent in sents]
    return sents, sents_words
