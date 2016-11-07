from collections import defaultdict
from collections import Counter
import preprocess as process
import utilities as utility
import math

def ngram_vector_keys(tokens, ngram_size=1):
  vector_keys = []
  for i in range(len(tokens)-(ngram_size-1)):
    vector_keys.append(" ".join(tokens[i:i+(ngram_size)]))
  return vector_keys

def JaccardCoefficient(sent_1_tokens, sent_2_tokens, ngram_size=1):
  set1 = set(ngram_vector_keys(sent_1_tokens, ngram_size))
  set2 = set(ngram_vector_keys(sent_2_tokens, ngram_size))
  return float(len(set1.intersection(set2)))/len(set1.union(set2))

def TFIDF(documents):
  Vocabulary = Counter()
  DocVectors = []
  IDFVector = Counter()
  No_of_Documents = float(len(documents))
  for document in documents:
    tf_single_doc_count = Counter(process.tokens(document))
    Vocabulary+= tf_single_doc_count
    DocVectors.append(tf_single_doc_count)
    IDFVector += Counter(tf_single_doc_count.keys())
  # print IDFVector
  for key in IDFVector.keys():
    IDFVector[key] = math.log(No_of_Documents/(1+IDFVector[key]))
  # print IDFVector
  # IDFVector = defaultdict(lambda:0.0, dict((key,Vocabulary[key]*) for key in c.keys()))
  TFIDFScores = defaultdict(lambda:0.0, dict((key,Vocabulary[key]*IDFVector[key]) for key in Vocabulary.keys()))
  # print TFIDFScores, IDFVector, Vocabulary
  return TFIDFScores, Vocabulary, DocVectors, IDFVector

def DocvectorTFIDF(TFIDFScores, tokens):
  return defaultdict(lambda:0.0, dict((key,TFIDFScores[key] if key in TFIDFScores.keys() else 0.0) for key in tokens))

def cosinesimilarity(document1, document2, TFIDFScores):
  tokens1 = set(process.tokens(document1))
  tokens2 = set(process.tokens(document2))
  vector1 = DocvectorTFIDF(TFIDFScores, tokens1)
  vector2 = DocvectorTFIDF(TFIDFScores, tokens2)
  len_vector_1 = math.sqrt(sum({k: v**2 for k, v in vector1.items()}.values()))
  len_vector_2 = math.sqrt(sum({k: v**2 for k, v in vector2.items()}.values()))
  cosine_similarity_score = (utility.dict_dotprod(vector1,vector2))/float((len_vector_2*len_vector_1))
  return cosine_similarity_score

def cosinesimilarity_without_TFIDF(document1, document2, TFIDFScores):
  vector1 = Counter(process.tokens(document1))
  vector2 = Counter(process.tokens(document2))
  len_vector_1 = math.sqrt(sum({k: v**2 for k, v in vector1.items()}.values()))
  len_vector_2 = math.sqrt(sum({k: v**2 for k, v in vector2.items()}.values()))
  cosine_similarity_score = (utility.dict_dotprod(vector1,vector2))/float((len_vector_2*len_vector_1))
  return cosine_similarity_score
