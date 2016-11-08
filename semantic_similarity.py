import lib.collect_data as data_collection
import math
import numpy as np

def main():
  Data = data_collection.Data();
  corpus_size = len(Data)
  print "The Total Corpus Data is about " + str(corpus_size)
  cosinesimilarity_without_TFIDF(Data)
  train_split = 0.8
  training_data_documents_size = math.round(corpus_size * train_split)
  test_data_documents_size = corpus_size - training_data_documents_size
  training_documents =  Data[:training_data_documents_size]
  test_documents = Data[training_data_documents_size+1:]
  print "Training on " + str(training_data_documents_size*2) + " documents"
  print "Testing on " + str(training_data_documents_size*2) + " documents"
  training_documents = [item[0], item[1] for item in training_documents]
  test_documents = [item[0], item[1] for item in test_documents]

def cosine_similarity_without_tfidf(documents):
  answers = []
  predicted_answers = []
  for document in documents:
    answers.append(document[2])
    predicted_answers.append(5 * cosinesimilarity_without_TFIDF(document[0], document[1]))
  evaluate(answers, predicted_answers)

def evaluate(gold_standard, predicted_answers):
  np.sum(np.power(np.array(gold_standard) - np.array(gold_standard),2))


def train(documents):
  ;

if __name__ == "__main__":
  main()