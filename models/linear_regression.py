from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import lib.utilities as utility
from sklearn import preprocessing

def linear_regression(training_documents, test_documents, training_answers,  test_answers):
  lm = LinearRegression()
  # doc_dict_vectors_list = []
  # Corpus = []
  # vectorizer = CountVectorizer(min_df=1)
  # for i in range(len(documents)/2):
  #   Corpus.append(documents[(2*i)] + " " + documents[(2*i)+1])  
  # X = vectorizer.fit_transform(Corpus)
  # vectorizer.transform(['Something completely new.']).toarray()
  # pX = v.fit_transform(D2)

  Doc_Dict_Vectors = utility.get_dict_vectors_of_documents(training_documents)
  # print Doc_Dict_Vectors
  v = DictVectorizer(sparse=True)

  X = v.fit_transform(Doc_Dict_Vectors)
  # print X
  # print training_answers
  min_max_scaler = preprocessing.MaxAbsScaler()
  X_train_minmax = min_max_scaler.fit_transform(X)
  X_normalized = preprocessing.normalize(X, norm='l2')
  lm.fit(X_train_minmax, training_answers)
  predicted_answers = lm.predict(X_train_minmax)
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
  print "Error in Estimation of Linear Regression - Training : "+str(utility.evaluate(training_answers,answers))
  Test_doc_dict_vectors = utility.get_dict_vectors_of_documents(test_documents)
  pX = v.transform(Test_doc_dict_vectors)
  pX_normalized = preprocessing.normalize(pX, norm='l2')
  pX_test_minmax = min_max_scaler.fit_transform(pX)
  predicted_answers = lm.predict(pX_test_minmax)
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
  print "Error in Estimation of Linear Regression - Testing : "+str(utility.evaluate(test_answers,answers))
