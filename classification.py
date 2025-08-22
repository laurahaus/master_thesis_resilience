# -*- coding: utf-8 -*-
"""
@author: Laura
"""

#%% import of important packages

import pandas as pd
import os  # to import data
import numpy as np
import spacy
import re
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, RandomizedSearchCV
import joblib
import itertools
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, make_scorer
import matplotlib.pyplot as plt
from clause_segmenter import ClauseSegmenter

nlp = spacy.load('en_core_web_lg')

os.chdir(r"C:\Users\Startklar\OneDrive\Dokumente\Master\Data")



#%% import of data

coded_seg = pd.read_excel('segments_data_cleaned.xlsx')
corpus_coded = pd.read_excel('10p_coded_data_sentences.xlsx')
corpus_uncoded = pd.read_excel('90p_uncoded_data_sentences.xlsx')

corpus_gold = pd.read_excel('gold_standard.xlsx')


#%% general functions for classification

# split training and test data
def stratifiedSplit(X, y, test_size = 0.25):
    """
    Splits data into training and test data while preserving the class distribution.

    Returns training and test data.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


# train classifier for relevant data
def trainClassifier(X_train, X_test, y_train, y_test, pipeline, param_grid, scoring='f1_macro', cv=5, average='macro'):
    """
    Tunes a pipline with GridSearchCV, evaluates the model and selects the best one.

    Returns the best estimator selected by grid search.

    """
    # define model settings
    pipe = pipeline
    param_grid = param_grid
    grid_search = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
    # grid_search = RandomizedSearchCV(pipe, param_grid, n_iter=200, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
    
    # perform grid search
    grid_search.fit(X_train, y_train)
    print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    best_model = grid_search.best_estimator_

    # evaluate the best model
    y_pred = best_model.predict(X_test)
    print('F1 score: {:.4f}'.format(f1_score(y_test, y_pred, average=average)))
    print('\nClassification Report: \n', classification_report(y_test, y_pred))
    
    return best_model


#%% relevance classifier

# determine optimal threshold for imbalanced data
def optimalThreshold(X_train, X_test, y_train, y_test, model, scoring, average, cv):
    """
    Tunes the decision threshold of a classifier with cross-validation, selects the best threshold and evaluates performance on the test set.
    
    Returns the optimal decision threshold.

    """    
    classifier_tuned = TunedThresholdClassifierCV(model, scoring=scoring, cv=cv, random_state=42).fit(X_train, y_train)
    best_th = classifier_tuned.best_threshold_
    print('Best threshold: {:.2f}'.format(best_th))
    
    # evaluate model performance
    y_pred = classifier_tuned.predict(X_test)
    print('F1 score: {:.4f}'.format(f1_score(y_test, y_pred, average=average)))
    print('\nClassification Report: \n', classification_report(y_test, y_pred))
    
    return best_th


# train relevance classifier
def trainRelevanceClassifier(X, y, pipeline, param_grid, test_size, scoring, cv, average):
    """
    Splits data into training and test set, trains a relevance classifier using cross-validated grid search, finds the optimal decision threshold
    and returns final model including the decision threshold.

    Returns the best model and its optimal decision threshold.

    """
    X_train, X_test, y_train, y_test = stratifiedSplit(X, y, test_size = test_size)
    model = trainClassifier(X_train, X_test, y_train, y_test, pipeline, param_grid, scoring=scoring, cv=cv, average=average)
    best_threshold = optimalThreshold(X_train, X_test, y_train, y_test, model, scoring=scoring, average=average, cv=cv)
    
    return model, best_threshold


def recallScoreWithConstraint(y_true, y_pred):
    """
    Computes recall score only if precision is at least 0.4, otherwise returns 0.

    Returns the constrained recall score.

    """
    precision = precision_score(y_true, y_pred)
    if precision < 0.4:
        return 0
    return recall_score(y_true, y_pred)


# extract relevant sentences 
def applyRelevanceClassifier(df, model, threshold):
    """
    Applies trained relevance classifier to identify relevant sentences.

    Returns a DataFrame containing only relevant sentences.

    """
    pred = model.predict_proba(df['cleaned_sentence'])[:, 1]
    pred_binary = (pred >= threshold).astype(int)
    df_pred = pd.DataFrame({'doc_no': df['doc_no'], 'cleaned_sentence': df['cleaned_sentence'], 'pred_label': pred_binary, 'original_sentence': df['original_sentence']})
    df_relevant = df_pred[df_pred['pred_label'] == 1]
    df_relevant = df_relevant.drop('pred_label', axis=1).reset_index(drop=True)
    return df_relevant


#%% general functions for data cleaning

# extract clauses 
def getClauses(original_sent):
    """
    Splits sentences or phrases into clauses. Returns the original sentence if it cannot be split.

    Returns list with all clauses.

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


# clean phrases
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


def mergeClauses(df):
    """
    Merges consecutive clauses from the same sentence that share the same predicted capacity label, ensuring cleaner outputs.

    Returns a DataFrame with merged clauses and their labels.

    """
    df_new = df.copy()
    df_unique = df_new[~df_new.duplicated(subset=['original_sentence', 'pred'], keep=False)]
    duplicates = df_new[df_new.duplicated(subset=['original_sentence', 'pred'], keep=False)]   
    unique_sentence = duplicates['original_sentence'].unique()
    
    df_merged = pd.DataFrame()
    
    for sent in unique_sentence:
        list_sent = duplicates.loc[duplicates['original_sentence'] == sent].copy()
        list_sent['start_idx'] = list_sent.apply(lambda row: row['original_sentence'].find(row['clause']), axis=1)
        list_sent['end_idx'] = list_sent.apply(lambda row: row['start_idx'] + len(row['clause']), axis=1)
        list_sent = list_sent.sort_values(by=['original_sentence', 'start_idx', 'pred']).reset_index(drop=True)
        for i in range(len(list_sent)-1, -1, -1):
            if i > 0 and list_sent.loc[i, 'pred'] == list_sent.loc[i-1, 'pred'] and list_sent.loc[i, 'start_idx'] - list_sent.loc[i-1, 'end_idx'] <= 3:
                list_sent.loc[i-1, 'clause'] += ' ' + list_sent.loc[i, 'clause']
                list_sent.loc[i-1, 'clean_clause'] += ' ' + list_sent.loc[i, 'clean_clause']
            else:
                df_merged = pd.concat([df_merged, list_sent.loc[i].to_frame().T], ignore_index=True)
    df_final = pd.concat([df_unique, df_merged])
    
    return df_final


#%% resilience capacity classifier

# histogram for threshold determination
def plotProbHistogram(X_test, y_test, model):
    """
    Predicts class probabilites and plots histograms showing distributions for correctly and incorrectly predicted samples.

    Displays a histogram per class label.
    
    """
    # determine probabilities and corresponding class labels
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    class_labels = model.classes_
    df_prob = pd.DataFrame(y_pred_prob, columns=['prob_{}'.format(label) for label in class_labels])
    df_label = pd.DataFrame({'y_real': y_test.to_list(),
                             'y_pred': y_pred})
    df_combined = pd.concat([df_label, df_prob], axis=1)
    
    # plot histogram for each label
    for label in class_labels:
        correct = df_combined.loc[(df_combined['y_real'] == label) & (df_combined['y_pred'] == label), 'prob_{}'.format(label)]
        incorrect = df_combined.loc[(df_combined['y_real'] == label) & (~df_combined['y_pred'].isin([label])), 'prob_{}'.format(label)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(correct, bins=20, alpha=0.5, color='blue', label="Correctly Predicted")
        plt.hist(incorrect, bins=10, alpha=0.5, color='red', label="Incorrectly Predicted")
        plt.xlabel("Probability of {} classification".format(label))
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(0, 1, 0.1))

        plt.show()
            

# train capacity classifier
def trainCapacityClassifier(X, y, pipeline, param_grid, test_size, scoring, cv, average):
    """
    Splits training and test data, trains a capacity classifier using cross-validated grid search, plots histograms to manually determine 
    the optimal decision thresholds for each class and returns final model.

    Returns the best-performing model fitted to the training data.
    """
    X_train, X_test, y_train, y_test = stratifiedSplit(X, y, test_size = test_size)
    model = trainClassifier(X_train, X_test, y_train, y_test, pipeline, param_grid, scoring=scoring, cv=cv, average=average)
    # plotProbHistogram(X_test, y_test, model)
    plotProbHistogram(X_train, y_train, model)
    
    return model


def precisionScoreWithConstraint(y_true, y_pred):
    """
    Computes precision score only if recall is at least 0.4, otherwise returns 0.

    Returns the constrained precision score.

    """
    recall = recall_score(y_true, y_pred, average='macro')
    if recall < 0.2:
        return 0
    return precision_score(y_true, y_pred, average='macro')

    
# final classification
def applyCapacityClassifier(df, segmenter, model, thresholds):
    """
    Splits relevant sentences into clauses, applies pretrained capacity classifier to clauses and filters capacities based on
    costum probability thresholds. 

    Returns a DataFrame containing clauses and their corresponding capacity.

    """
    df_new = df.copy()
    # split relevant data into clauses
    df_new['clause'] = df_new['original_sentence'].apply(getClauses)
    df_new = df_new.explode('clause')
    df_new = df_new.dropna(subset=['clause'])
    df_new['clean_clause'] = df_new['clause'].apply(cleanPhrase)
    df_new = df_new.reset_index(drop=True)
        
    # make predictions
    pred = model.predict(df_new['clean_clause'])
    prob = model.predict_proba(df_new['clean_clause'])
    class_labels = model.classes_
    
    # create df with probabilities
    df_prob = pd.DataFrame(prob, columns=['prob_{}'.format(label) for label in class_labels])
    df_pred = pd.DataFrame({'doc_no': df_new['doc_no'],
                            'pred': pred,
                            'clause': df_new['clause'],
                            'clean_clause': df_new['clean_clause'],
                            'original_sentence': df_new['original_sentence']})
    df_combined = pd.concat([df_pred, df_prob], axis=1)
    
    # filter classes
    df_final = pd.DataFrame()
    for label in class_labels:
        mask = df_combined['prob_{}'.format(label)] > thresholds[label]
        df_relevant = df_combined[mask]
        df_final = pd.concat([df_final, df_relevant])
    df_clauses_capacity = df_final.drop_duplicates()
    
    return df_clauses_capacity
   

#%% apply relevance classification model

# parameter for model training
pipe_1 = Pipeline([('vectorizer', CountVectorizer()),
                   ('transformer', TfidfTransformer()),
                   ('over', SMOTE(random_state=42)),
                   ('under', RandomUnderSampler(random_state=42)),
                   ('cb', ComplementNB())])


corpus_coded['label_bin'] = corpus_coded['label'].map({'irrelevant': 0, 'relevant': 1})

param_grid_1 = {'vectorizer__ngram_range': [(1,1), (1,2)],
                'vectorizer__max_features': [5700, 5800, 5900],
                'over__k_neighbors': [6, 7],
                'over__sampling_strategy': [{1: r} for r in range(2300, 2501, 100)],
                'under__sampling_strategy': [{0: i} for i in range(3500, 3701, 100)],
                'cb__alpha': [8, 9]
                }


recall_scorer = make_scorer(recallScoreWithConstraint)

# model training
model_relevance, threshold_relevance = trainRelevanceClassifier(corpus_coded['cleaned_sentence'], corpus_coded['label_bin'], pipe_1, param_grid_1, test_size=0.25, scoring=recall_scorer, cv = 10, average='binary')

# save model
joblib.dump(model_relevance, 'relevance_model.pkl')

model_relevance = joblib.load('relevance_model.pkl')


#%% apply capacity classifier

# parameter for model training
pipe_2 = Pipeline([('vectorizer', CountVectorizer()), 
                  ('transformer', TfidfTransformer()),
                  ('smote', SMOTE(random_state = 42)),  
                  ('lr', LogisticRegression(solver='saga', class_weight='balanced', random_state=20))])

param_grid_2 = {'vectorizer__ngram_range': [(1,4), (1,5)],
              'transformer__sublinear_tf': [True, False],
              'smote__k_neighbors': [10, 12],
              'smote__sampling_strategy': [{'adaptive': a, 'transformative': t} for a, t in itertools.product(range(370, 411, 20), range(380, 421, 20))],
              'lr__tol': [0.00001, 0.0001, 0.001],
              'lr__max_iter': np.arange(210, 291, 20)
              }

precision_scorer = make_scorer(precisionScoreWithConstraint)
    
model_capacity = trainCapacityClassifier(coded_seg['cleaned_code'], coded_seg['capacity'], pipe_2, param_grid_2, test_size=0.3, scoring=precision_scorer, cv=5, average='macro')


# save model
joblib.dump(model_capacity, 'capacity_model.pkl')

model_capacity = joblib.load('capacity_model.pkl')


# manually determine thresholds from histograms
thresholds_capacity = dict(preventive = 0.41,
                           anticipative = 0.41,
                           absorptive = 0.39,
                           adaptive = 0.38,
                           transformative = 0.35) #0.35


#%% gold corpus

# calculate gold corpus
gold_relevant = applyRelevanceClassifier(corpus_coded, model_relevance, threshold_relevance)

segmenter = ClauseSegmenter(pipeline=nlp)

gold_relevant_clause = applyCapacityClassifier(gold_relevant, segmenter, model_capacity, thresholds_capacity)

gold_corpus_subset = corpus_gold[['doc_no', 'clause', 'capacity']].rename(columns={'capacity': 'real_capacity'})
gold_relevant_clause_subset = gold_relevant_clause[['doc_no', 'clause', 'pred']].rename(columns={'pred': 'pred_capacity'})

df_gold = pd.merge(gold_corpus_subset, gold_relevant_clause_subset, how = 'left', on = ('doc_no', 'clause'))
df_gold['pred_capacity'] = df_gold['pred_capacity'].fillna('irrelevant')

print('\nClassification Report: \n', classification_report(df_gold['real_capacity'], df_gold['pred_capacity']))


# calculate confusion matrix
cm = confusion_matrix(df_gold['real_capacity'], df_gold['pred_capacity'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


#%% apply classifiers

# apply relevance model
corpus_uncoded_relevant = applyRelevanceClassifier(corpus_uncoded, model_relevance, threshold_relevance)

# apply capacity model
corpus_uncoded_capacity = applyCapacityClassifier(corpus_uncoded_relevant, segmenter, model_capacity, thresholds_capacity)
corpus_uncoded_capacity_merged = mergeClauses(corpus_uncoded_capacity)

corpus_uncoded_predictions = corpus_uncoded_capacity_merged[['doc_no', 'pred', 'clause', 'clean_clause']]

corpus_uncoded_predictions.to_excel('corpus_uncoded_predictions.xlsx', index=False)