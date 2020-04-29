import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import enchant
from spellchecker import SpellChecker
import time
from nltk.corpus import wordnet
import textstat


def remove_special_char(essay):
    return re.sub('@\S+|[^A-Za-z0-9]',' ',essay)

def word_tokenizer(essay):
    return nltk.word_tokenize(essay)

def sent_tokenizer(essay):
    return nltk.sent_tokenize(essay)

def count(tokens):
    return len(tokens)

def punct_count(essay):
    count = 0
    punctuations = string.punctuation
    for char in essay:
        if char in punctuations:
            count += 1
    return count

def check_spell(words):
    d = enchant.Dict("en_US")
    spell = SpellChecker()
    misspelled = set()
    err_count = 0
    for word in words:
        if d.check(word) == False:
            misspelled.add(word)
            err_count += 1
    corr_dict = {}
    for word in misspelled:
        corr_dict[word] = spell.correction(word)
    essay_df = pd.DataFrame(words)
    essay_df.replace(corr_dict,inplace=True)
    essay = ' '.join(list(essay_df[0]))
    return err_count,essay

def create_documents(essay):
    stop_words=set(stopwords.words('english'))
    stop_words.remove('not')
    lemmatizer = WordNetLemmatizer()
    essay = essay.lower()
    essay = nltk.word_tokenize(essay)
    essay=[lemmatizer.lemmatize(word) for word in essay if not word in stop_words]
    essay=' '.join(essay)
    return essay

def pos_count(tokens):
    noun_count = 0
    verb_count = 0
    adv_count = 0
    adj_count = 0
    word_pos = nltk.pos_tag(tokens)
    for pos in word_pos:
        if pos[1][0] == 'N':
            noun_count += 1
        elif pos[1][0] == 'V':
            verb_count += 1
        elif pos[1][0] == 'J':
            adj_count += 1
        elif pos[1][0] == 'R':
            adv_count += 1
    return noun_count, verb_count, adv_count, adj_count


def compute_redability(essay):
    return textstat.flesch_reading_ease(essay)


def unique_word_prop(tokens):
    ratio = len(set(tokens))/len(tokens)
    return ratio


def get_synonyms(tokens):
    synonyms = set()
    for word in tokens:
        synset = nltk.wordnet.wordnet.synsets(word)
        for ss in synset:
            for swords in ss.lemma_names():
                synonyms.add(swords.lower())
    return list(synonyms)


def nouns_and_verbs_pos(tokens):
    word_pos = nltk.pos_tag(tokens)
    nouns_and_verbs = set()
    for pos in word_pos:
        if (pos[1][0] == 'V') | (pos[1][0] == 'N'):
            nouns_and_verbs.add(pos[0])
    return list(nouns_and_verbs)