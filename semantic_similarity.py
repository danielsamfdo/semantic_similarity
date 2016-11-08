import lib.collect_data as data_collection
import models.ngram as ngram
import math as math
import itertools
import numpy as np

def main():
  Data = data_collection.Data();
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
  #Char_IDFVector = ngram.CharacterIDFVector(train_documents, ngram_size=2)
  print "Training Documents Analysis"
  train_predicted_answers = cosinesimilarity_evaluate_TFIDF(train_documents, TFIDFScores, training_documents_answers)
  print "Error in Estimate is " + str(evaluate(training_documents_answers, train_predicted_answers))
  # print "Test Documents Analysis"
  # cosinesimilarity_evaluate_TFIDF(test_documents, TFIDFScores, test_documents_answers)

def cosinesimilarity_evaluate_TFIDF(documents, TFIDFScores, answers):
  ind = 0
  predicted_answers = []
  for i in range(len(documents)/2):
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    print document1, document2
    predicted_answers.append(cosinesimilarity(document1, document2, TFIDFScores))
    print answers[ind],   "   |   ", predicted_answers[ind]
    ind+=1
  return predicted_answers 


if __name__ == "__main__":
  main()