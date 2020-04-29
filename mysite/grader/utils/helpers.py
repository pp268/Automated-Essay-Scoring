import string
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

def create_data(essay,essay_set,essay_prompt_df,essay_source_df,vectorizer):

    clean_essay = remove_special_char(essay)
    word_tokens = word_tokenizer(clean_essay)
    sent_tokens = sent_tokenizer(essay)
    word_count = count(word_tokens)
    sent_count = count(sent_tokens)
    spell_err , corrected_essay = check_spell(word_tokens)
    essay_documents = create_documents(corrected_essay)
    corrected_tokens = word_tokenizer(essay_documents)
    noun_count,verb_count,adv_count,adj_count = pos_count(corrected_tokens)
    readability_score = compute_redability(essay)
    unique_word_ratio = unique_word_prop(corrected_tokens)
    
    if essay_set in [1,2,7,8]:
        prompt_tokens = essay_prompt_df.loc[essay_prompt_df['essay_set'] == essay_set,'tokens'][essay_set]
        synonyms = essay_prompt_df.loc[essay_prompt_df['essay_set'] == essay_set,'synonyms'][essay_set]
        synonyms_overlap_temp = [word for word in corrected_tokens if word in synonyms]
        prompt_overlap_temp = [word for word in corrected_tokens if word in prompt_tokens]
        synonyms_overlap = (len(synonyms_overlap_temp))
        synonyms_overlap_prop = (len(synonyms_overlap_temp)/(len(corrected_tokens)+1))
        prompt_overlap = (len(prompt_overlap_temp))
        prompt_overlap_prop = (len(prompt_overlap_temp)/(len(corrected_tokens)+1))
        X = pd.DataFrame({'word_count' : word_count, 'sent_count' : sent_count, 'spell_err' : spell_err,'noun_count': noun_count,
                        'verb_count' : verb_count,'adv_count':adv_count, 'adj_count':adj_count, 'readability_score':readability_score, 
                        'unique_word_ratio':unique_word_ratio,'syn_overlap':synonyms_overlap, 'syn_overlap_prop':synonyms_overlap_prop,
                        'prompt_overlap':prompt_overlap,'prompt_overlap_prop':prompt_overlap_prop},index=[0])
    
    
    else:
        source_tokens = essay_source_df.loc[essay_source_df['essay_set'] == essay_set,'pos(nouns & verbs)'][essay_set]
        source_overlap_temp = [word for word in corrected_tokens if word in source_tokens]
        source_overlap = (len(source_overlap_temp))
        source_overlap_prop = (len(source_overlap_temp)/(len(corrected_tokens)+1))
        synonyms = essay_source_df.loc[essay_source_df['essay_set'] == essay_set,'synonyms'][essay_set]
        synonyms_overlap_temp = [word for word in corrected_tokens if word in synonyms]    
        synonyms_overlap = (len(synonyms_overlap_temp))
        synonyms_overlap_prop = (len(synonyms_overlap_temp)/(len(corrected_tokens)+1))
        X = pd.DataFrame({'word_count' : word_count, 'sent_count' : sent_count, 'spell_err' : spell_err,'noun_count': noun_count,
                        'verb_count' : verb_count,'adv_count':adv_count, 'adj_count':adj_count, 'readability_score':readability_score, 
                        'unique_word_ratio':unique_word_ratio,'syn_overlap':synonyms_overlap, 'syn_overlap_prop':synonyms_overlap_prop,
                        'source_overlap':source_overlap,'source_overlap_prop':source_overlap_prop},index=[0])
    vectors = vectorizer.transform([essay_documents])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist() 
    df = pd.DataFrame(denselist, columns=feature_names)
    
    final_df = pd.concat([X,df],axis=1)
    
    return final_df