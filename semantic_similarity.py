import lib.collect_data as data_collection
import models.ngram as ngram
import models.linear_regression as lr
import models.svm as svm
import models.neuralnet_MLPRegressor as mlp
import math as math
import itertools
import numpy as np
import lib.preprocess as process
import lib.utilities as utility
import pdb
import models.w2vec as w2vec

def main():
  Data = data_collection.Data();
  Data = Data[:10]
  corpus_size = len(Data)
  print "The Total Corpus Data is about " + str(corpus_size*2)
  cosine_similarity_without_tfidfy_without_tfidf_predicted_answers = cosine_similarity_without_tfidf(Data)
  train_split = 0.8
  training_data_documents_size = int(round(corpus_size * train_split))
  #print training_data_documents_size
  test_data_documents_size = corpus_size - training_data_documents_size
  training_documents =  Data[:training_data_documents_size]
  # print training_data_documents_size
  test_documents = Data[training_data_documents_size:]
  # print test_documents
  print "Training on " + str(training_data_documents_size*2) + " documents"
  print "Testing on " + str(test_data_documents_size*2) + " documents"
  training_documents_answers = [item[2] for item in training_documents]
  training_documents = [item[0:2] for item in training_documents]
  training_documents = list(itertools.chain.from_iterable(training_documents))
  test_documents_answers = [item[2] for item in test_documents]
  test_documents = [item[0:2] for item in test_documents]
  test_documents = list(itertools.chain.from_iterable(test_documents))
  w2vec_model = w2vec.w2vec_model()
  # print training_documents_answers
  # train_and_test_simple_model(training_documents, test_documents, training_documents_answers, test_documents_answers, load=True)
  train_and_test_Linear_Regression_model(training_documents, test_documents, training_documents_answers, test_documents_answers, load=False, w2vec_model=w2vec_model, use_w2_vec_model=True)
  # train_and_test_SVM_model(training_documents, test_documents, training_documents_answers, test_documents_answers, load=True, w2vec_model=w2vec_model, use_w2_vec_model=True)
  # train_and_test_MLP_model(training_documents, test_documents, training_documents_answers, test_documents_answers, load=True, w2vec_model=w2vec_model, use_w2_vec_model=True)
  # w2vec.w2vec_similarity_measure_unsupervised(training_documents+test_documents, training_documents_answers+test_documents_answers)

def cosine_similarity_without_tfidf(documents):
  answers = []
  predicted_answers = []
  ind=0
  for document in documents:
    answers.append(document[2])
    predicted_answers.append(5 * ngram.cosinesimilarity_without_TFIDF(document[0], document[1]))
    # print answers[ind],   "   |   ",  predicted_answers[ind]
    ind+=1
  print "Error in Estimate is " + str(utility.evaluate(answers, predicted_answers))
  return predicted_answers

def evaluate(gold_standard, predicted_answers):
  return np.sum(np.absolute(np.array(gold_standard) - np.array(predicted_answers)))/len(gold_standard) 
  # return math.sqrt(np.sum(np.power(np.array(gold_standard) - np.array(predicted_answers),2))) 

def train_and_test_simple_model(train_documents, test_documents, training_documents_answers, test_documents_answers, load=False):
  if(not load):
    TFIDFScores, Vocabulary, DocVectors, IDFVector = ngram.TFIDF(train_documents)
    #print TFIDFScores
    utility.save_weights("TFIDFScores.dat",TFIDFScores);
    utility.save_weights("Vocabulary.dat", Vocabulary);
    utility.save_weights("DocVectors.dat", DocVectors);
    utility.save_weights("IDFVector.dat", IDFVector);
  else:
    TFIDFScores=utility.load_weights("weights/TFIDFScores.dat")
    #print TFIDFScores
    Vocabulary=utility.load_weights("weights/Vocabulary.dat")
    DocVectors=utility.load_weights("weights/DocVectors.dat")
    IDFVector=utility.load_weights("weights/IDFVector.dat")
  f = open("analysis.txt","w")
  # print "Training Documents Analysis"
  # print "-------------------------------------------------------------------"
  # train_predicted_answers = cosinesimilarity_evaluate_TFIDF(train_documents, TFIDFScores, training_documents_answers)
  
  # print "Test Documents Analysis"
  # print "-------------------------------------------------------------------"
  # test_predicted_answers = cosinesimilarity_evaluate_TFIDF(test_documents, TFIDFScores, test_documents_answers)
  for ngram_size in range(1,5):
    for analyze_type in ["lemma","pos","character"]:#, "character"]:
      if(analyze_type=="character"):
        if(not load):
          idf_scores = ngram.CharacterIDFVector(train_documents, ngram_size)
          utility.save_weights("IDF_Char_"+str(ngram_size)+"_gram.dat", idf_scores)
        else:
          idf_scores = utility.load_weights("weights/IDF_Char_"+str(ngram_size)+"_gram.dat")
      else:
        idf_scores = IDFVector
      # print "-------------------------------------------------------------------"
      # f.write("-------------------------------------------------------------------")
      # print "Jaccards and Containment Coefficient Analysis without using ngram weighing and type = " + analyze_type + " ngram = " + str(ngram_size)
      # f.write("Jaccards and Containment Coefficient Analysis without using ngram weighing and type = " + analyze_type + " ngram = " + str(ngram_size))
      # print "-------------------------------------------------------------------"
      # f.write("-------------------------------------------------------------------")
      # Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, train_documents, training_documents_answers, ngram_size, False, None,f)
      # Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, test_documents, test_documents_answers, ngram_size, False, None,f)
      print "-------------------------------------------------------------------"
      f.write("-------------------------------------------------------------------")
      print "Jaccards and Containment Coefficient Analysis using ngram weighing and type = " + analyze_type + " ngram = " + str(ngram_size)
      f.write("Jaccards and Containment Coefficient Analysis using ngram weighing and type = " + analyze_type + " ngram = " + str(ngram_size))
      print "-------------------------------------------------------------------"
      f.write("-------------------------------------------------------------------")
      Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, train_documents, training_documents_answers, ngram_size, True, idf_scores,f)
      Jacc_one_gram_pred_answers, Containment_one_gram_pred_answers = jaccard_and_containment_coefficient_evaluate(analyze_type, test_documents, test_documents_answers, ngram_size, True, idf_scores,f)
  f.close()

def cosinesimilarity_evaluate_TFIDF(documents, TFIDFScores, answers):
  ind = 0
  predicted_answers = []
  for i in range(len(documents)/2):
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    #print document1, document2
    predicted_answers.append(5*ngram.cosinesimilarity(document1, document2, TFIDFScores))
    #print answers[ind],   "   |   ", predicted_answers[ind]
    ind+=1
  print "Error in Estimate is " + str(utility.evaluate(answers, predicted_answers))
  return predicted_answers 

def jaccard_and_containment_coefficient_evaluate(analyze_type, documents, answers, ngram_size=1, ngram_weighing=False, IDFScores=None,f=None):
  ind = 0
  containment_coefficient_predicted_answers = []
  jaccard_coefficient_predicted_answers = []
  init_doc_count = len(documents)/2
  operated_doc_count = 0
  for i in range(len(documents)/2):
    if(operated_doc_count == 400):
      init_doc_count-=400
      operated_doc_count=0
      print str(init_doc_count) + " sets remaining"
    operated_doc_count+=1
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    #print document1, document2
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
    #print answers[ind],   "   |   ", jaccard_coefficient_predicted_answers[ind]
    #print answers[ind],   "   |   ", containment_coefficient_predicted_answers[ind]
    ind+=1
  print "Error in Estimate For Jaccard Coefficient is " + str(utility.evaluate(answers, jaccard_coefficient_predicted_answers))
  f.write("Error in Estimate For Jaccard Coefficient is " + str(utility.evaluate(answers, jaccard_coefficient_predicted_answers)))
  print "Error in Estimate For Containment Coefficient is " + str(utility.evaluate(answers, containment_coefficient_predicted_answers))
  f.write("Error in Estimate For Containment Coefficient is " + str(utility.evaluate(answers, containment_coefficient_predicted_answers)))
  return jaccard_coefficient_predicted_answers, containment_coefficient_predicted_answers

def train_and_test_Linear_Regression_model(train_documents, test_documents, training_documents_answers, test_documents_answers, load=False, w2vec_model=None, use_w2_vec_model=True):
  lr.linear_regression(train_documents,  test_documents, training_documents_answers, test_documents_answers, load=load, w2vec_model=w2vec_model, use_w2_vec_model=True)
  return

def train_and_test_SVM_model(train_documents, test_documents, training_documents_answers, test_documents_answers, load=False, w2vec_model=None, use_w2_vec_model=True):
  svm.support_vector_machines(train_documents,  test_documents, training_documents_answers, test_documents_answers, load=load, w2vec_model=w2vec_model, use_w2_vec_model=True)
  return

def train_and_test_MLP_model(train_documents, test_documents, training_documents_answers, test_documents_answers, load=False, w2vec_model=None, use_w2_vec_model=True):
  mlp.mlp_network(train_documents,  test_documents, training_documents_answers, test_documents_answers, load=load, w2vec_model=w2vec_model, use_w2_vec_model=True)
  return

if __name__ == "__main__":
  main()
