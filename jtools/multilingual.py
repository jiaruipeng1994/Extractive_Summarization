#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : Mon Mar 28 10:40:22 2022
# @Author : JRP - Ruipeng Jia

##################################################################
## For MLSUM
from spacy.lang.de import German
from spacy.lang.fr import French
from spacy.lang.es import Spanish
from spacy.lang.tr import Turkish
from spacy.lang.ru import Russian
spacy_tokenizer = {'de': German(), 'fr': French(), 'es': Spanish(), 'tr': Turkish(), 'ru': Russian()}
