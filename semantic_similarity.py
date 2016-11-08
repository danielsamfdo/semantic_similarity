import lib.collect_data as data_collection
import models.ngram as ngram
import math as math
import itertools
import numpy as np
import lib.preprocess as process

def main():
  Data = data_collection.Data();
  #Data = Data[:10]
  corpus_size = len(Data)
  print "The Total Corpus Data is about " + str(corpus_size)
  #cosine_similarity_without_tfidf_predicted_answers = cosine_similarity_without_tfidf(Data)
  train_split = 0.8
  training_data_documents_size = int(round(corpus_size * train_split))
  #print training_data_documents_size
  test_data_documents_size = corpus_size - training_data_documents_size
  training_documents =  Data[:training_data_documents_size]
  test_documents = Data[training_data_documents_size+1:]
  print "Training on " + str(training_data_documents_size*2) + " documents"
  print "Testing on " + str(test_data_documents_size) + " documents"
  training_documents_answers = [item[2] for item in training_documents]
  training_documents = [item[0:2] for item in training_documents]
  training_documents = list(itertools.chain.from_iterable(training_documents))
  test_documents_answers = [item[2] for item in test_documents]
  test_documents = [item[0:2] for item in test_documents]
  test_documents = list(itertools.chain.from_iterable(test_documents))
  train_and_test_simple_model(training_documents, test_documents, training_documents_answers, test_documents_answers)


def cosine_similarity_without_tfidf(documents):
  answers = []
  predicted_answers = []
  ind=0
  for document in documents:
    answers.append(document[2])
    predicted_answers.append(5 * ngram.cosinesimilarity_without_TFIDF(document[0], document[1]))
    print answers[ind],   "   |   ",  predicted_answers[ind]
    ind+=1
  print "Error in Estimate is " + str(evaluate(answers, predicted_answers))
  return predicted_answers

def evaluate(gold_standard, predicted_answers):
  return math.sqrt(np.sum(np.power(np.array(gold_standard) - np.array(predicted_answers),2))) 

def train_and_test_simple_model(train_documents, test_documents, training_documents_answers, test_documents_answers):
  TFIDFScores, Vocabulary, DocVectors, IDFVector = ngram.TFIDF(train_documents)
  #print "Came ine "
  Char_IDFVector = ngram.CharacterIDFVector(train_documents, ngram_size=2)
  print "Training Documents Analysis"
  print "-------------------------------------------------------------------"
  train_predicted_answers = cosinesimilarity_evaluate_TFIDF(train_documents, TFIDFScores, training_documents_answers)
  
  print "Test Documents Analysis"
  print "-------------------------------------------------------------------"
  test_predicted_answers = cosinesimilarity_evaluate_TFIDF(test_documents, TFIDFScores, test_documents_answers)
  for ngram_size in range(1,2):
    for analyze_type in ["pos", "lemma", "character"]:
      if(analyze_type=="character"):
        idf_scores = Char_IDFVector
      else:
        idf_scores = IDFVector
      print "-------------------------------------------------------------------"  
      print "Jaccards and Containment Coefficient Analysis using ngram weighing and type = " + analyze_type + " ngram = " + str(ngram_size)
      print "-------------------------------------------------------------------"
      Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, train_documents, training_documents_answers, ngram_size, False, IDFScores=None)
      Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, test_documents, test_documents_answers, ngram_size, False, IDFScores=None)
      print "-------------------------------------------------------------------"  
      print "Jaccards and Containment Coefficient Analysis using ngram weighing and type = " + analyze_type + " ngram = " + str(ngram_size)
      print "-------------------------------------------------------------------"  
      Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, train_documents, training_documents_answers, ngram_size, True, idf_scores)
      Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, test_documents, test_documents_answers, ngram_size, True, idf_scores)
    
def cosinesimilarity_evaluate_TFIDF(documents, TFIDFScores, answers):
  ind = 0
  predicted_answers = []
  for i in range(len(documents)/2):
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    print document1, document2
    predicted_answers.append(5*ngram.cosinesimilarity(document1, document2, TFIDFScores))
    print answers[ind],   "   |   ", predicted_answers[ind]
    ind+=1
  print "Error in Estimate is " + str(evaluate(answers, predicted_answers))
  return predicted_answers 

def jaccard_and_containment_coefficient_evaluate(analyze_type, documents, answers, ngram_size=1, ngram_weighing=False, IDFScores=None):
  ind = 0
  containment_coefficient_predicted_answers = []
  jaccard_coefficient_predicted_answers = []
  for i in range(len(documents)/2):
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    print document1, document2
    sent_1_tokens = process.tokens(document1)
    sent_2_tokens = process.tokens(document2)
    if(analyze_type=="pos"):
      jaccard_coefficient, containment_coefficient = ngram.POSTags_JaccardCoefficient_and_containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size, ngram_weighing, IDFScores)
    elif(analyze_type=="lemma"):
      jaccard_coefficient, containment_coefficient = ngram.Lemma_JaccardCoefficient_and_containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size, ngram_weighing, IDFScores)
    elif(analyze_type=="character"):
      jaccard_coefficient, containment_coefficient = ngram.character_ngram_JaccardCoefficient_and_containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size, True, ngram_weighing, IDFScores)
    jaccard_coefficient_predicted_answers.append(5*jaccard_coefficient)
    containment_coefficient_predicted_answers.append(5*containment_coefficient)
    print answers[ind],   "   |   ", jaccard_coefficient_predicted_answers[ind]
    print answers[ind],   "   |   ", containment_coefficient_predicted_answers[ind]
    ind+=1
  print "Error in Estimate For Jaccard Coefficient is " + str(evaluate(answers, jaccard_coefficient_predicted_answers))
  print "Error in Estimate For Containment Coefficient is " + str(evaluate(answers, containment_coefficient_predicted_answers))
  return jaccard_coefficient_predicted_answers, containment_coefficient_predicted_answers

if __name__ == "__main__":
  main()