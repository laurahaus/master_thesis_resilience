# -*- coding: utf-8 -*-
"""
@author: Laura
"""

#%% import of important packages

import pandas as pd
import os  # to import data
from collections import defaultdict
import spacy
from parse import parse
import re
import numpy as np
from clause_segmenter import ClauseSegmenter
from fuzzywuzzy import fuzz, process

# working directory
os.chdir(r"C:\Users\Startklar\OneDrive\Dokumente\Master\Data")

# spacy model
nlp = spacy.load('en_core_web_lg')


#%% functions for creating df

# import txt files, extract all the relevant data and save them in df
def txtToDf(directory):
    """
    Imports txt files from local directory and extracts release year, disaster type and document number from filenames.
    
    Returns DataFrame with all important information for further analysis.

    """
    
    dic = defaultdict(list) # create dictionary, which automatically creates new key
    
    # read txt files
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            file_name = os.path.splitext(file)[0]
            dic['name'].append(file_name)
            with open(os.path.join(directory, file), encoding='utf-8') as f:
                lines = filter(None, (line.rstrip() for line in f))
                for line in lines:
                    (key, val) = line.split(':', 1)
                    #print(key, val)
                    dic[key].append(val)
    df = pd.DataFrame(dic)
    
    # extract important information
    df.rename(columns={'yy-mm-dd': 'date'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['date'])
    df['year'] = df['datetime'].dt.year
    df = extractInformation(df)
    
    # clean df
    df_grouped = df.groupby('doc_no')['disaster'].apply(list).reset_index()
    df = df.drop(columns=['name', 'ART_ID3', 'disaster', 'datetime'])
    df_unique = df.drop_duplicates()
    df_final = df_unique.merge(df_grouped, on='doc_no', how='left')
    return df_final

# extract document number and disaster type for articles
def extractInformation(df):
    """
    Extracts disaster type and document number from filenames.

    Returns DataFrame with separate columns for disaster types and document number.

    """
    pattern = 'AllUK2000-2023_{rest}_{disaster}_{doc}'
    doc_no = []
    disaster = []
    for file in df['name']:
        result = parse(pattern, file)
        doc_no.append(result['doc'])
        disaster.append(result['disaster'])
    df['doc_no'] = doc_no
    df['disaster'] = disaster
    return df

# delete articles that are in 10p and 90p corpora & add disaster types to 10p
def deleteDuplicateArticles(df1, df2):
    """
    Remove articles from the 90p corpus that are in both the 10p & 90p corpora. Disaster types mentioned in each article are saved in a list in 10p corpus.

    Returns the 10p and 90p corpus with filtered duplicate articles.

    """
    doc_df1 = df1['doc_no'].unique()
    doc_df2 = df2['doc_no'].unique()
    common_elements = np.intersect1d(doc_df1, doc_df2)
    df2_filtered = df2[~df2.doc_no.isin(common_elements)]
    df1_combined = df1.copy()
    for doc in common_elements:
        list1 = df1.loc[df1['doc_no'] == doc, 'disaster'].values[0]
        list2 = df2.loc[df2['doc_no'] == doc, 'disaster'].values[0]
        list_combined = list1 + list2
        df1_combined.at[df1[df1['doc_no'] == doc].index[0], 'disaster'] = list_combined
    return df1_combined, df2_filtered


# create look up dictionary where all sentences for each document are saved
def createDictionary(df):
    """
    Splits text into sentences and builds a look up dictionary that saves the document number as the key and each individual sentence of this document. 

    Returns a dictionary with each document number and a list of corresponding sentences.

    """
    dic = dict()
    for n in range(len(df)):
        text = nlp(df.loc[n, 'Bodytext'])
        dic[df.loc[n, 'doc_no']] = list(text.sents)       
    return dic

# extract entire sentence for each phrase
def extractSentences(df, dic):
    """
    Matches each annotated clause to its corresponding sentence using the look up dictionary. Removes special characters for consistent comparison 
    in case some characters are not displayed correctly.

    Returns a DataFrame with a new column 'sentence' containing the corresponding sentence for each phrase.

    """
    df = df.copy()
    df['sentence'] = None
    pattern = re.compile(r'[^\w\s\.\,]+')
    for n in range(len(df)): 
        doc = df.loc[n, 'doc_no']
        list_sentences = dic[doc]
        for sents in list_sentences:
            code_clean = pattern.sub('', df.loc[n, 'code'])
            sentences_clean = pattern.sub('', str(sents))
            if code_clean in sentences_clean:
                df.loc[n, 'sentence'] = str(sents)
    return df


#%% functions for cleaning text 

# general data cleaning 
def preprocessing(text):
    """
    Cleans data by removing mail addresses, digits, fixed expressions, non-alphabetic characters and frequent disaster types,
    then lowercases and normalises spacing.

    Returns the cleaned text as a string.

    """
    text = re.sub(r'\S*@\S*\s?', '', text) # remove emails
    text = re.sub(r'\w*\d\w*', '', text) # remove digits in text
    text = re.sub(r'\'', '', text)  # remove apastrophes
    text = re.sub(r'(abstract|full text)', '', text, flags=re.IGNORECASE) # remove start sentences
    text = re.sub(r'[^\w\d\s]', ' ', text) # remove non-alphabetic characters
    frequent_words = ['flood', 'drought', 'heatwave', 'wildfire', 'hurricane', 'tornado', 'storm']
    for word in frequent_words:
        text = re.sub(r'\b' + word + r's?\b', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text

# clean phrases for annotated examples
def cleanPhrase(text):
    """
    Applies preprocessing function and tokenisation, stop word removal and lemmatization to annotated clauses.

    Returns cleaned annotated clauses as a string.

    """
    prepro_text = preprocessing(text)
    with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'ner', 'lemmatizer']):
        doc = nlp(prepro_text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.ent_type_]
        clean_text = ' '.join(tokens)
    return clean_text

# clean articles for corpora; includes sentence segmentation
def createCleanedDf(df):
    """
    Splits each article into sentences, cleans them and saves them with the corresponding document number and the original sentence.
    
    Returns a DataFrame containing all original and cleaned sentence including their document number.

    """
    df_new = df.copy()
    df_new['original_sentence'], df_new['cleaned_sentence'] = segmentSentences(df_new['Bodytext'])
    df_sample = df_new[['doc_no', 'original_sentence', 'cleaned_sentence']].copy()
    df_sentences = df_sample.explode(['original_sentence', 'cleaned_sentence']).reset_index(drop=True)
    df_sentences_cleaned = df_sentences.drop(df_sentences[df_sentences['cleaned_sentence'].str.strip() == ''].index)
    return df_sentences_cleaned

def segmentSentences(texts):
    """
    Splits texts into sentences and applies preprocessing steps to each sentence.
    
    Returns two lists: one with the original sentences the other one with their cleaned counterparts.

    """
    original_sentences = []
    cleaned_sentence = []
    with nlp.select_pipes(enable=['tok2vec', 'parser']):
        for doc in nlp.pipe(texts, batch_size=20):
            list_sents = [sent.text for sent in doc.sents]
            original_sentences.append(list_sents)
            
    with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'ner', 'lemmatizer']):
        for doc_sents in original_sentences:
            cleaned_sents = []
            cleaned_doc = [preprocessing(sent) for sent in doc_sents]
            for doc in nlp.pipe(cleaned_doc, batch_size=20):
                tokens = [token.lemma_ for token in doc if not token.ent_type_]
                cleaned_sents.append(" ".join(tokens))
            cleaned_sentence.append(cleaned_sents)
    return original_sentences, cleaned_sentence


#%% functions for creating gold standard dataset

def labelRelevantArticles(df, df_relevant):
    """
    Determines for each sentence in the smaller subsample corpus if it containes relevant information, by checking if one of the
    annotated phrases is part of the sentence

    Returns a DataFrame where each sentence is classified as relevant or irrelevant.

    """
    df_new = df.copy()
    set_df = set(zip(df_new['doc_no'], df_new['original_sentence']))
    set_relevant = set(zip(df_relevant['doc_no'], df_relevant['sentence']))
    common_items = set_relevant.intersection(set_df)
    df_new['label'] = df_new.apply(lambda row: 'relevant' if (row['doc_no'], row['original_sentence']) in common_items else 'irrelevant', axis=1)
    return df_new


def createGoldStandard(df_sentence, df_segments, segmenter):
    """
    Creates gold standard dataset by segmenting each sentence of the smaller subsample corpus into clauses and match these clauses
    with the annotated phrases.

    Returns a DataFrame with clause-level labels (gold standard).

    """
    # get clauses for 10p
    df_sent = df_sentence.copy()
    df_sent['clause'] = df_sent['original_sentence'].apply(getClauses)
    df_sent = df_sent.explode('clause').reset_index(drop=True)

    # get clauses for annotated phrases
    df_seg = df_segments.copy()
    df_seg['code_clause'] = df_seg['code'].apply(getClauses)
    df_seg = df_seg.explode('code_clause').reset_index(drop=True)
    
    df_sent['capacity'] = 'irrelevant'
        
    for doc in df_seg['doc_no'].unique():
        list_sents = df_sent.loc[((df_sent['label'] == 'relevant') & (df_sent['doc_no'] == doc)), 'clause']
        df_seg_subset = df_seg.loc[df_seg['doc_no'] == doc].reset_index(drop=True)
        for idx_seg, seg in enumerate(df_seg_subset['code_clause']):
            sent = process.extract(seg, list_sents, scorer = fuzz.partial_ratio, limit = 1)[0][0]
            idx_sent = df_sent.index[(df_sent['label'] == 'relevant') & (df_sent['doc_no'] == doc) & (df_sent['clause'] == sent)][0]
            df_sent.loc[idx_sent, 'capacity'] = df_seg_subset.loc[idx_seg, 'capacity']
    return df_sent

# clean clauses & remove duplicates 
def getClauses(original_sent):
    """
    Uses a clause segmenter to split sentences or segments into clauses. Returns the original sentence if it cannot be split.

    Returns a list with all clauses.

    """
    clauses = segmenter.get_clauses_as_list(original_sent)
    if len(clauses) == 0:
        return [original_sent]
    else:
        sorted_clauses = sorted(clauses, key=len, reverse=True)
        unique_clauses = []
        for clause in sorted_clauses:
            if not any(clause in existing_clause for existing_clause in unique_clauses):
                unique_clauses.append(clause)
        return unique_clauses


#%% import txt files & coded segments 

# import coded & uncoded corpora
df_coded = txtToDf('sample_laura_2025_new')
df_uncoded = txtToDf('sample_laura_2025_90p')

# delete duplicates & assign disaster types from uncoded corpus to coded corpus
df_coded, df_uncoded = deleteDuplicateArticles(df_coded, df_uncoded)

# import coded segments and add relevant information
coded_seg = pd.read_excel('coded_segments.xlsx')
coded_seg = coded_seg.rename(columns={'Dokumentname': 'name', 'Segment': 'code', 'Code-Alias': 'capacity'})
extractInformation(coded_seg)

# save dfs
df_coded.to_excel('10p_coded_data.xlsx', index=False)
df_uncoded.to_excel('90p_uncoded_data.xlsx', index=False)


#%% extract sentences for annotated phrases

# create look up dictionary and extract sentences
dic_sentences = createDictionary(df_coded)
coded_seg = extractSentences(coded_seg, dic_sentences)
coded_seg = coded_seg.dropna(subset=['sentence']).reset_index(drop=True) # delete 3 phrases that could not be found


#%% data cleaning 

df_sent_coded = createCleanedDf(df_coded)
df_sent_uncoded = createCleanedDf(df_uncoded)

coded_seg['cleaned_code'] = coded_seg['code'].apply(cleanPhrase)
coded_seg = coded_seg.drop_duplicates(subset=['doc_no', 'cleaned_code', 'capacity']).reset_index(drop=True)


#%% create gold standard dataset

# determine relevant segments
df_sent_coded = labelRelevantArticles(df_sent_coded, coded_seg)

df_sent_coded.to_excel('10p_coded_data_sentences.xlsx', index=False)
coded_seg.to_excel('segments_data_cleaned.xlsx', index=False)
df_sent_uncoded.to_excel('90p_uncoded_data_sentences.xlsx', index=False)

# create gold standard
segmenter = ClauseSegmenter(pipeline=nlp)
df_gold = createGoldStandard(df_sent_coded, coded_seg, segmenter)

df_gold.to_excel('gold_standard.xlsx', index=False)
