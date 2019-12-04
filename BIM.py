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




def make_inverted_index(articles):
    """
    This function builds an inverted index as an hash table (dictionary)
    where the keys are the terms and the values are ordered lists of
    docIDs containing the term.
    """
    articles = remove_stop_words(articles)
    index = defaultdict(set)
    for docid, article in enumerate(articles):
        for term in article:
            index[term].add(docid)
    return index


# ## Transform documents and queries in boolean vectors



def term_document_matrix(articles):
    """
    Given a set of articles we generate a term-document matrix 
    with value 1 if the term is in the document and 0 otherwise
    """
    index = make_inverted_index(articles)
    terms = list(index.keys())
    mat = np.zeros((len(terms), len(articles)))
    for i, t in enumerate(terms):
        posting_list = index[t]
        for j in posting_list:
            mat[i][j] = 1
    return mat




def query_to_vector(query_string, index):
    '''
    This function transforms a free form text query into a boolean vector 
    with i-th value 1 if the the i-th term in the index is present in the query
    zero otherwise 
    '''
    query = query_string.upper().split()
    terms = index.keys()
    query_vector = np.zeros((len(terms),))
    for i,t in enumerate(terms):
        if t in query:
            query_vector[i] = 1
    return query_vector


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
    


# ## Ranking function


def RSV_dq(doc_vector, query_vector, weights):
    '''
    This function computes the RSV for a given couple document - query 
    using the precomputed weights
    '''
    tot = 0
    for i,term in enumerate(weights.keys()):
        tot += doc_vector[i] * query_vector[i] * weights[term]
    return tot


# ## BIM Class


class BIM():
    '''
    Binary Independence Model class
    '''
    
    def __init__(self, corpus):
        self.articles = corpus
        self.index = make_inverted_index(corpus)
        self.term_doc_matrix = term_document_matrix(corpus)
        self.weights = RSV_weights(self.articles, self.index)
        self.ranked = []
        self.query_text = ''
        
        
    def ranking(self, query):
        '''
        Auxiliary function for the function answer query. Computes the scores for each
        document in the corpus using the precomputed weights.
        '''
        
        query_vector = query_to_vector(query, self.index)
        
        scores = []
        for i in range(len(self.articles)):
            scores.append((i, RSV_dq(self.term_doc_matrix[:,i], query_vector, self.weights)))
        
        self.ranked = sorted(scores, key=lambda x: x[1], reverse = True)
        return self.ranked
    
    
    def answer_query(self, query):
        '''
        Function to answer a free text query. Shows the first 30 words of the
        15 most relevant documents.
        '''
        
        self.query_text = query
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



