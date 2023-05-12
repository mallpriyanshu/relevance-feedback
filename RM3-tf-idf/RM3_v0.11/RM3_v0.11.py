# %%
import math
import lucene
import time
import itertools
import numpy as np
from tqdm import tqdm
from java.io import File
import xml.etree.ElementTree as ET
from collections import defaultdict
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, TermQuery, BoostQuery
from org.apache.lucene.search.similarities import BM25Similarity, LMJelinekMercerSimilarity, LMDirichletSimilarity
lucene.initVM()

# %%
q_name = 'trec6'

# %%
index_path = '../index/'
topicFilePath = f'../{q_name}.xml'

directory = FSDirectory.open(File(index_path).toPath())
indexReader = DirectoryReader.open(directory)

# %%
def query_topics(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    parsed_topics = {}

    for top in root.findall('top'):
        num = top.find('num').text.strip()
        title = top.find('title').text.strip()
        parsed_topics[num] = title

    return parsed_topics

# %%
query_all = query_topics(topicFilePath)

# %%
def getDocumentVector(luceneDocid, indexReader):

    N = indexReader.numDocs()                   
    
    docVec = defaultdict(lambda: [0, 0]) 
    D = 0                                 
    
    terms = indexReader.getTermVector(luceneDocid, 'CONTENTS')
    iterator = terms.iterator()
    for term in BytesRefIterator.cast_(iterator):
        t = term.utf8ToString()
        tf = iterator.totalTermFreq()  
        df = indexReader.docFreq(Term('CONTENTS', t))  
        D += tf
        docVec[t][0] = tf
        docVec[t][1] = df
    
    docVec = {key: (value[0] / D) * math.log(N / (value[1] + 1)) for key, value in docVec.items()}
    
    total_weight = sum(docVec.values())
    docVec = {key: value / total_weight for key, value in docVec.items()}

    # print(len(docVec), "D", D)
    return docVec, D


# %%
# terms = indexReader.getTermVector(1, 'CONTENTS')
# terms.getStats()

# %%
def search(indexReader, query, similarity, top_rel_doc):
    analyzer = EnglishAnalyzer()
    searcher = IndexSearcher(indexReader)
    searcher.setSimilarity(similarity)
    query = QueryParser("CONTENTS", analyzer).parse(query)

    scoreDocs = searcher.search(query, top_rel_doc).scoreDocs
    
    docids = [scoreDoc.doc for scoreDoc in scoreDocs]

    set_cont = set()
    set_cont = {term for doc in docids for term in getDocumentVector(doc, indexReader)[0].keys()}

    filtered_tok = set()
    for tok in set_cont:
        if tok.isalpha():
            filtered_tok.add(tok)

    # N = indexReader.numDocs()  
    # new_set = []
    # for t in set_cont:
    #     df = indexReader.docFreq(Term('CONTENTS', t)) 
    #     if df/N < 0.1:
    #         new_set.append(t)
            
    # print('Old Set:', len(set_cont))
    # print('New Set:', len(new_set))

    # return set_cont, docids
    return filtered_tok, docids

# %%
# def RM3_term_selection(Query, set_ET, docs, indexReader, alpha, mu):
    
#     totalTF = indexReader.getSumTotalTermFreq("CONTENTS")

#     Q = Query.split()
#     weight = {}

#     cf = {}
#     for t in set_ET:
#         T = Term("CONTENTS", t)
#         cf[t] = indexReader.totalTermFreq(T)/totalTF
#     for q in Q:
#         set_ET.add(q)
#         T = Term("CONTENTS", q)
#         cf[q] = indexReader.totalTermFreq(T)/totalTF

#     docVectors = {}
#     mixinglambda = {}
#     doclength = {}
    
#     for d in docs:                    
#         docVectors[d], doclength[d] = getDocumentVector(d, indexReader)
        
#     for d in docs:                  
#         mixinglambda[d] = doclength[d]/(doclength[d] + mu)
        
#     for w in set_ET:
#         p_wr = 0
#         for d in docs:                  
#             ml = mixinglambda[d]
#             p_wd = (ml*(docVectors[d].get(w,0)) + (1 - ml)*cf[w])      
#             # p_wd = (docVectors[d].get(w,0))      

#             p_q = 1
#             for q in Q:
#                 p_q = p_q*(ml*(docVectors[d].get(q,0)) + (1 - ml)*cf[q])          

#             p_wr = p_wr + p_wd*p_q
#         weight[w] = p_wr

#     norm = sum(weight.values())
    
#     if norm == 0:
#         print(Q,'\n\n')
#     else:
#         weight = {w:weight[w]/norm for w in weight}

#     for w in set_ET:
#         weight[w] = (alpha*weight[w]) + (1-alpha)*(Q.count(w)/len(Q))

#     temp_list = sorted(weight.items(), key=lambda x:x[1], reverse=True)
#     sorted_weights = dict(temp_list)

#     return sorted_weights

# %%
def RM3_term_selection(Query, set_ET, docs, indexReader, alpha, mu, expanded_query_terms):
    
    totalTF = indexReader.getSumTotalTermFreq("CONTENTS")

    Q = Query.split()
    weight = {}

    cf = {}
    for t in set_ET | set(Q):
        T = Term("CONTENTS", t)
        cf[t] = indexReader.totalTermFreq(T)/totalTF

    docVectors = {}
    mixinglambda = {}
    doclength = {}
    
    for d in docs:                    
        docVectors[d], doclength[d] = getDocumentVector(d, indexReader)
        
    for d in docs:                  
        mixinglambda[d] = doclength[d]/(doclength[d] + mu)
        
    for w in set_ET:
        p_wr = 0
        for d in docs:                  
            ml = mixinglambda[d]
            # p_wd = (ml*(docVectors[d].get(w,0)) + (1 - ml)*cf[w]) 
            p_wd = docVectors[d].get(w,0)     
        
            p_q = 1
            for q in Q:
                # p_q = p_q*docVectors[d].get(q,0)   
                      
                p_q = p_q*(ml*(docVectors[d].get(q,0)) + (1 - ml)*cf[q])   


            p_wr = p_wr + p_wd*p_q
        weight[w] = p_wr



    weight = dict(sorted(weight.items(), key=lambda x:x[1], reverse=True)[:expanded_query_terms])
    
    norm = sum(weight.values())
    if norm == 0:
        pass
    else:
        weight = {w:weight[w]/norm for w in weight}


 
    for w in weight.keys() | set(Q):
        weight[w] = (alpha*weight.get(w,0)) + (1-alpha)*(Q.count(w)/len(Q))
   

    temp_list = sorted(weight.items(), key=lambda x:x[1], reverse=True)
    sorted_weights = dict(temp_list)

    return sorted_weights

# %%
# def expanded_query_BM25(search, RM3_term_selection, k1, b, alpha, top_rel_doc, expanded_query_terms, mu):

#     analyzer = EnglishAnalyzer()
#     similarity = BM25Similarity(k1,b)
#     expanded_q = []

#     i = 0
#     for q in tqdm(query_all.values(), colour='red', desc='Expanding Queries'):
#     # for q in query_all.values():
     
#         i += 1 
#         escaped_q = QueryParser('CONTENTS', analyzer).escape(q)      # a few titles had '/' in them which 
#         query = QueryParser('CONTENTS', analyzer).parse(escaped_q)
        
#         query_terms = [term.strip()[9:] for term in query.toString().split()]
#         parsed_q = ' '.join(query_terms)
# #         print(parsed_q)
        
#         expension_term_set, docids = search(indexReader, parsed_q, similarity, top_rel_doc)
#         # expension_term_set, docids = search(indexReader, query, similarity, top_rel_doc)

#         weights = RM3_term_selection(parsed_q, expension_term_set, docids, indexReader, alpha, mu)
#         query_len = len(query_terms)
#         # query_len = 0
        
#         expanded_query_terms_list = list(weights.keys())[0:expanded_query_terms + query_len]
#         expanded_query_w = list(weights.values())[0:expanded_query_terms + query_len]
        
#         norm = sum(expanded_query_w)
#         expanded_query_weights = list(np.array(expanded_query_w)/norm)
    
#         booleanQuery = BooleanQuery.Builder()
#         for m in range(expanded_query_terms + query_len):
#             t = Term('CONTENTS', expanded_query_terms_list[m])
#             tq = TermQuery(t)
#             boostedTermQuery = BoostQuery(tq, float(expanded_query_weights[m]))
#             BooleanQuery.setMaxClauseCount(4096)
#             booleanQuery.add(boostedTermQuery, BooleanClause.Occur.SHOULD)
#         booleanQuery = booleanQuery.build()
       
#         expanded_q.append(booleanQuery)   

#     return expanded_q

# %%
def expanded_query_BM25(search, RM3_term_selection, k1, b, alpha, top_rel_doc, expanded_query_terms, mu):

    analyzer = EnglishAnalyzer()
    similarity = BM25Similarity(k1,b)
    expanded_q = []

    i = 0
    # for q in tqdm(query_all.values(), colour='red', desc='Expanding Queries'):
    for q in query_all.values():
     
        i += 1 
        escaped_q = QueryParser('CONTENTS', analyzer).escape(q)      # a few titles had '/' in them which 
        query = QueryParser('CONTENTS', analyzer).parse(escaped_q)
        
        query_terms = [term.strip()[9:] for term in query.toString().split()]
        parsed_q = ' '.join(query_terms)
#         
        
        expension_term_set, docids = search(indexReader, parsed_q, similarity, top_rel_doc)
        # expension_term_set, docids = search(indexReader, query, similarity, top_rel_doc)
        weights = RM3_term_selection(parsed_q, expension_term_set, docids, indexReader, alpha, mu, expanded_query_terms)
        # print(weights)
    
        booleanQuery = BooleanQuery.Builder()
        for m, n in weights.items():
            t = Term('CONTENTS', m)
            tq = TermQuery(t)
            boostedTermQuery = BoostQuery(tq, float(n))
            BooleanQuery.setMaxClauseCount(4096)
            booleanQuery.add(boostedTermQuery, BooleanClause.Occur.SHOULD)
        booleanQuery = booleanQuery.build()
       
        expanded_q.append(booleanQuery)   

    return expanded_q

# %%
def search_retrived(indexReader, Query, Qid, similarity, out_name):

    searcher = IndexSearcher(indexReader)
    searcher.setSimilarity(similarity)
   
    scoreDocs = searcher.search(Query, 1000).scoreDocs             #retrieving top 1000 relDoc
    i = 1
    res = ''

    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        r = str(Qid) + '\t' + 'Q0' + '\t' + str(doc.get('ID')) + '\t' + str(i) + '\t' + str(scoreDoc.score) + '\t' + str(out_name) + '\n'
        res += r
        i = i+1   

    return res

# %%
def run_RM3(top_PRD, expanded_query_terms, alpha, mu):
    expand_q = expanded_query_BM25(search, RM3_term_selection, k1, b, alpha, top_PRD, expanded_query_terms, mu)
                                       
    name = 'prm_'
    sim = BM25Similarity(k1,b)
    name = name + 'BM25_' + str(k1) + '_'+ str(b)

    file_name = f'./res_RM3/{q_name}/fixed_mu_lambda/{q_name}_mu_' + str(mu) +'_docs_' + str(top_PRD) + '_terms_' + str(expanded_query_terms) + '_alpha_' + str(alpha) + '.txt'
    out_file = open(file_name, "w")

    res = ''
    for i in tqdm(range(len(query_all)),colour='cyan', desc = 'Re-retrival'):
    # for i in range(len(query_all)):
    
        result =  search_retrived(indexReader, expand_q[i], list(query_all.keys())[i], sim, name)
        res = res + result

    out_file.write(res)
    out_file.close()
    # print("Retrieval Completed - result dumped in", file_name)

# %%
k1 = 0.8
b = 0.4

top_PRD = range(5,51,5)
expanded_query_terms = range(0, 101, 5)
alpha = [0.7]
mu = [750]

parameters = list(itertools.product(top_PRD, expanded_query_terms, alpha, mu))

for num_doc, num_q, alpha, mu in tqdm(parameters, colour='red'):
    run_RM3(num_doc, num_q, alpha, mu)

# %%



