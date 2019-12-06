from collections import defaultdict
from math import log, sqrt
import re
import numpy as np
import time
import sys


# ## Import the data ad create the inverted index

def import_dataset():
    """
    This function import all the articles in the TIME corpus,
    returning list of lists where each sub-list contains all the
    terms present in the document as a string.
    """
    articles = []
    with open('TIME.ALL', 'r') as f:
        tmp = []
        for row in f:
            if row.startswith("*TEXT"):
                if tmp != []:
                    articles.append(tmp)
                tmp = []
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                tmp += row.split()
    return articles




def remove_stop_words(corpus):
   '''
   This function removes from the corpus all the stop words present in the file TIME.STP
   '''    
   
   stop_w = [line.rstrip('\n') for line in open('TIME.STP')]
   stop_w=list(filter(None, stop_w))
   for i in range(0,len(corpus)):
       corpus[i] = [x for x in corpus[i] if x not in stop_w]
   return corpus 




def make_inverted_index(corpus):
    """
    This function builds an inverted index as an hash table (dictionary)
    where the keys are the terms and the values are ordered lists of
    docIDs containing the term.
    """
    corpus = remove_stop_words(corpus)
    index = defaultdict(set)
    for docid, article in enumerate(corpus):
        for term in article:
            index[term].add(docid)
    return index



def posting_lists_union(pl1, pl2):
    """
    Returns a new posting list resulting from the union of this
    one and the one passed as argument.
    """
    pl1 = sorted(list(pl1))
    pl2 = sorted(list(pl2))
    union = []
    i = 0
    j = 0
    while (i < len(pl1) and j < len(pl2)):
        if (pl1[i] == pl2[j]):
            union.append(pl1[i])
            i += 1
            j += 1
        elif (pl1[i] < pl2[j]):
            union.append(pl1[i])
            i += 1
        else:
            union.append(pl2[j])
            j += 1
    for k in range(i, len(pl1)):
        union.append(pl1[k])
    for k in range(j, len(pl2)):
        union.append(pl2[k])
    return union


# ## Precomputing weights


def DF(term, index):
    return len(index[term])



def IDF(term, index, corpus):
    return log(len(corpus)/DF(term, index))



def RSV_weights(corpus,index):
    '''
    This function precomputes the Retrieval Status Value weights 
    for each term in the index
    '''
    N = len(corpus)
    w = {}
    for term in index.keys():
        p = DF(term, index)/(N+0.5)  
        w[term] = IDF(term, index, corpus) + log(p/(1-p))
    return w
    

# ## BIM Class


class BIM():
    '''
    Binary Independence Model class
    '''
    
    def __init__(self, corpus):
        self.articles = corpus
        self.index = make_inverted_index(corpus)
        self.weights = RSV_weights(self.articles, self.index)
        self.ranked = []
        self.query_text = ''
    
    
    def RSV_doc_query(self, doc_id, query):
        '''
        This function computes the RSV for a given couple document - query
        using the precomputed weights
        '''
        score = 0
        doc = self.articles[doc_id]
        for term in doc:
            if term in query:
                score += self.weights[term]
        return score
    
    
    def ranking(self, query):
        '''
        Auxiliary function for the function answer query. Computes the score only
        for documents that are in the posting list of al least one term in the query
        '''
        
        docs = []
        for term in self.index:
            if term in query:
                docs = posting_lists_union(docs, self.index[term])
    
        scores = []
        for doc in docs:
            scores.append((doc, self.RSV_doc_query(doc, query)))

        self.ranked = sorted(scores, key=lambda x: x[1], reverse = True)
        return self.ranked
    
    
    
    def answer_query(self, query_text):
        '''
        Function to answer a free text query. Shows the first 30 words of the
        15 most relevant documents.
        '''
        
        self.query_text = query_text
        query =  query_text.upper().split()
        ranking = self.ranking(query)
        
        for i in range(0, 15):
            article = self.articles[ranking[i][0]]
            if (len(article) > 30):
                article = article[0:30]
            text = " ".join(article)
            print(f"Article {i + 1}, score: {ranking[i][1]}")
            print(text, '\n')
            
            
    def relevance_feedback(self, *args):
        '''
        Function that implements relevance feedback for the last query answered.
        The weights are recomputed based on a set of relevant documents provided by the user
        '''
        if(self.query_text == ''):
            sys.exit('Cannot get feedback before a query is formulated.')
        
        relevant_idx = list(args)
        
        if(isinstance(relevant_idx[0], list)):
            relevant_idx = relevant_idx[0]
        
        relevant_docs = []
        for idx in relevant_idx:
            doc_id = self.ranked[idx-1][0]
            relevant_docs.append(self.articles[doc_id])
        
        N = len(self.articles)
        N_rel = len(relevant_idx)
        
        for term in self.index.keys():
            vri = 0
            for doc in relevant_docs:
                if term in doc:
                    vri += 1
            p = (vri + 0.5) /( N_rel + 1)
            u = (DF(term, self.index) - vri + 0.5) / (N - N_rel +1) 
            self.weights[term] = log((1-u)/u) + log(p/(1-p))
        
        self.answer_query(self.query_text)
        

            

# Example of usage

# articles = import_dataset()
# bim  = BIM(articles)
# bim.answer_query('Italy and Great Britain fight the enemy')
# bim.relevance_feedback(5,6,8)



