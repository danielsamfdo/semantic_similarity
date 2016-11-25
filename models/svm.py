from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import lib.utilities as utility
from sklearn import preprocessing
import numpy as np
import sklearn

def test_hyper_parameters(X_total,y_total):
  C_range = np.logspace(-2, 10, 2)
  gamma_range = np.logspace(-9, 3, 2)
  param_grid = dict(gamma=gamma_range, C=C_range)
  grid = GridSearchCV(svm.SVR(), param_grid=param_grid, cv=5)
  grid.fit(X_total, y_total)
  print("The best parameters are %s with a score of %0.2f"  % (grid.best_params_, grid.best_score_))


def support_vector_machines(training_documents, test_documents, training_answers,  test_answers, load, w2vec_model, use_w2_vec_model):
  
  
  # doc_dict_vectors_list = []
  # Corpus = []
  # vectorizer = CountVectorizer(min_df=1)
  # for i in range(len(documents)/2):
  #   Corpus.append(documents[(2*i)] + " " + documents[(2*i)+1])  
  # X = vectorizer.fit_transform(Corpus)
  # vectorizer.transform(['Something completely new.']).toarray()
  # pX = v.fit_transform(D2)



  

  if(load):
    Doc_Dict_Vectors = utility.load_weights("weights/Feature_Vector.dat")
    Test_doc_dict_vectors = utility.load_weights("weights/Test_Feature_Vector.dat")
  else:
    Doc_Dict_Vectors = utility.get_dict_vectors_of_documents(training_documents)
    Test_doc_dict_vectors = utility.get_dict_vectors_of_documents(test_documents)
    utility.save_weights("Feature_Vector.dat",Doc_Dict_Vectors)
    utility.save_weights("Test_Feature_Vector.dat",Test_doc_dict_vectors)
  # print Doc_Dict_Vectors
  v = DictVectorizer(sparse=True)

  X = v.fit_transform(Doc_Dict_Vectors)
  

  ######### CODE FOR TESTING HYPER PARAMETERS ##################
  # Total_Doc_Dict_Vectors = utility.get_dict_vectors_of_documents(training_documents+test_documents)
  # X_total = v.fit_transform(Total_Doc_Dict_Vectors)
  # y_total = training_answers+test_answers
  # test_hyper_parameters(X_total,y_total)
  ##############################################################

  # print X
  # print training_answers
  min_max_scaler = preprocessing.StandardScaler(with_mean=False)
  X_train_minmax = min_max_scaler.fit_transform(X)
  X_normalized = preprocessing.normalize(X, norm='l2')
  C=10000000000.0
  gamma = 1000
  print "Trying C = %s,  Gamma = %s"%(str(C), str(gamma)) 
  svm_model = svm.SVR(C=C, gamma=gamma)
  svm_model.fit(X_train_minmax, training_answers)
  predicted_answers = svm_model.predict(X_train_minmax)
  answers = []
  for i in predicted_answers:
    if(i<0):
      # print "came in"
      answers.append(0)
    elif(i>5):
      # print "came in *** "
      answers.append(5)
    else:
      answers.append(i)
  # print answers
  print "Error in Estimation of SVM - Training : "+str(utility.evaluate(training_answers,answers))
  pX = v.transform(Test_doc_dict_vectors)
  pX_normalized = preprocessing.normalize(pX, norm='l2')
  pX_test_minmax = min_max_scaler.fit_transform(pX)
  predicted_answers = svm_model.predict(pX_test_minmax)
  answers = []
  for i in predicted_answers:
    if(i<0):
      # print "came in"
      answers.append(0)
    elif(i>5):
      # print "came in *** "
      answers.append(5)
    else:
      answers.append(i)
  # print answers
  print "Error in Estimation of SVM - Testing : "+str(utility.evaluate(test_answers,answers))
