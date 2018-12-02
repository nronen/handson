# fetch MNIST -- dataset (70,000 ) small images of handwritten digits

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X,y = mnist["data"],mnist["target"]
# drawing one of the images
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest" )
plt.axis("off")
plt.show()

# The MNIST database is already split into a training set (the first 60,000 images) and a test set (the last
# 10,000 images)
X_train, X_test, y_train, y_test = X[:60000],X[60000:],y[:60000],y[60000:]

# reshuffling the training set
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

## Training a binary classifier ( 5-detector )
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5  = (y_test == 5)

# Stochastic Gradient Descent (SGD)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train,y_train_5)

sgd_clf.predict([some_digit])

## Performance Measures

## Cross validation (similar to Sci-Kit cross_val_score() )
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Stratified sampling to produce folds that contain a representative ratio of each class
skfolds = StratifiedKFold(n_splits = 3 , random_state = 42)

# At each iteration the code creates a clone of the classifier, trains that clone on the Training
# folds and makes predictions on the test fold
for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds , y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print (n_correct/len(y_pred))

# The same process using cross_val_score()
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf,X_train, y_train_5, cv=3, scoring="accuracy"))

# Confusion Matrix : how many times instances of class A are classified as class B
# To commpute the confusion matrix, you need to have a set of predictions so they can be compared to
# the actual targets

from sklearn.model_selection import cross_val_predict
# The function cross_val_predict performs K-fold cross-validation, but instead of returning the evaluation
# scores it returns the predictions made on each test fold
# This means that you get a clean prediction for each instance in the training set ("clean") meaning
# that the prediction is made by a model that never saw the data during training

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix
# Each row in the confusion matrix represents an actual class, while each column represents a predicted class
confusion_matrix(y_train_5,y_train_pred)

#           predicted 0	      predicted 1
#           (Negative Class)  (Positive Class)
#actual 0 	True Negative	  False Positive
#actual 1 	False Negative    True Positive
#
# precision - TP / (TP + FP) ; recall = TP / (TP + FN)
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5,y_train_pred)
recall_score(y_train_5,y_train_pred)

# F1 score = 2PR/(P+R)
from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)

# For each instance, SGDClassifier computes a score based on decision function and if that score is greater than
# a threshold it assigns the instance to the positive class, else it assigns the instance to the Negative
# classself.
# Sci-kit learn does not let you set the threshold directly but it gives you access to the decision scores that
# it uses to make predictions. Instead of calling predict(),  you can call its decision_function() and make
# predictions based on those scores using any threshold you want :
y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_scores > threshold)

# SGDClassifier uses a threshold equal to 0. Raising the threshold decreases recall
# How to determine what is the right threshold ? For this you will need first to get the scores of all instances
# using cross_val_predict(), but this time specifying that you want to return decision scores instead of predictions :
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# With these scores you can compute precision and recall for all thresholds using the precision_recall_curve() function:
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds, recalls[:-1],"g--",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# The receiver operating characteristic (ROC) is another common tool used with binary classifiers
# The ROC plots True Positive rate (recall = TP/(TP+FN) ) against the False Positive rate (FP/(FP+TN) )
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5,y_scores)

def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr,tpr)
plt.show()

# One way to compare classifiers is to measure the area under the curve (AUC). A perfect classifier will
# have ROC-AUC equal 1, whereas a purely random classifier will have an ROC AUC eqaul 0.5

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

# As a rule of thumb, you should prefer the PR (precision/recall) curve whenever the positive class is rare or
# when you care more about false positives than the false negativesself.

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3 ,
                                    method = "predict_proba")

y_scores_forest = y_probas_forest[:,1] # score = proba of positive class
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show()

# The randomForestClassifier ROC curve looks much better than the SGDClassifier. As a result, its ROC AUC
# is significantly better
roc_auc_score(y_train_5, y_scores_forest)

y_train_forest_pred = cross_val_predict(forest_clf,X_train,y_train_5,cv=3)
precision_score(y_train_5,y_train_forest_pred)
recall_score(y_train_5,y_train_forest_pred)

# Multi-class classification using binary classifiers
# Two strategies - one-versus-all (OvA) and one-versus-one (OvO)
# OvA : for detecting digits, train 10 classifiers (one for each digit) and select the class whose classifier
# outputs the highest score
# OvO : Train a binary classifier for every pair of digits -> If there are N classes , you need (1/2)*N*(N-1) classifiers(!)
# the main advantage is that each classifier needs to be trained on the part of the training set for the two classes
# it must distinguish

# Under the hood, Scikit-learn trained 10 binary classifiers, got their decision scores and selected the class with
# the highest score. decision_function() returns 10 scores per class :
sgd_clf.fit(X_train,y_train)
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

# Note that when a classifier is trained, it stores the list of target classes in classes_ attribute
# Some algorithms (e.g. SVM) scales poorly with the size of the training set, in this case OvO is preferred. For most
# of the algorithms OvA is preferred

# Sci-kit learn, detects when you try to use binary classifier for multiclass classification task and it
# automatically runs OvA (except SVM class for which it uses OvO)

# Forcing Sci-kit learn to use OvA (OneVsRestClassifier()) or OvO (OneVsOneClassifier()) :
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train,y_train)
ovo_clf.predict([some_digit])

# As described earlier, scaling the inputs will improve the results :
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# Error Analysis - analysing the error that your model makes
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3 )
conf_mx = confusion_matrix(y_train, y_train_pred)

# Normalize the results (Divide each value by the number of images in the corresponding class)
row_sums = conf_mx.sum(axis=1, keepdims = True)
norm_conf_mx = conf_mx / row_sums

# Fill the diagonal with zeros to keep only the errors
np.fill_diagonal(norm_conf_mx,0)
# Image representation of the confusion Matrix (dark : good precition, bright : bad prediction)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
# 3/5 confusion
cl_a , cl_b = 3,5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25],images_per_row = 5)
plt.subplot(222); plot_digits(X_ab[:25],images_per_row = 5)
plt.subplot(223); plot_digits(X_ba[:25],images_per_row = 5)
plt.subplot(224); plot_digits(X_bb[:25],images_per_row = 5)
plt.show()

# Multilabel classification - the following code creates two target labels for each image :
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)

# One way to evaluate the performance of a multilabel classifier is to compute F1 score for each
# individual label and then to compute the average score :
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_pred, average = "macro")
# In this case all the labels are equally important. If you want to give each label a weight equal
# to its support (i.e. the number of instances with that target label), set average="weighted" in
# the preceding code
