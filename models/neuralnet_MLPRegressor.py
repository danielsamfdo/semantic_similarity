from sklearn import neural_network
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import lib.utilities as utility
from sklearn import preprocessing
import numpy as np
import sklearn

def test_hyper_parameters(X_total, y_total):
  ##### TO UPDATE #########
  alpha = 0.001
  learning_rate = 1


def mlp_network(training_documents, test_documents, training_answers,  test_answers, load, w2vec_model, use_w2_vec_model):

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
  mlp = neural_network.MLPRegressor(verbose=True,max_iter=1)
  mlp.fit(X_train_minmax, training_answers)
  predicted_answers = mlp.predict(X_train_minmax)
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
  print "Error in Estimation of MLP - Training : "+str(utility.evaluate(training_answers,answers))
  pX = v.transform(Test_doc_dict_vectors)
  pX_normalized = preprocessing.normalize(pX, norm='l2')
  pX_test_minmax = min_max_scaler.fit_transform(pX)
  predicted_answers = mlp.predict(pX_test_minmax)
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
  print "Error in Estimation of MLP - Testing : "+str(utility.evaluate(test_answers,answers))
