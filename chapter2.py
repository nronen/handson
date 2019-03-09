import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from zlib import crc32

# California Housing Prices dataset from StatLib repository

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# fetch the data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    os.chdir(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,"housing.tgz")
    os.chdir("../..")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# load the data using pandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def split_train_set(data,test_ratio) :
    np.random.seed(42) # to make sure we get the same test set every time
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio )
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# compute a hash of each instance's identifier, keep only the last byte of the hash
# select only the instances that their id is lower than 256 * test_ratio
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
    
# main
fetch_housing_data()
housing = load_housing_data()
# Get a quick description of the data
# housing.info()
#
# find out what categories exist and how many districts belong to each category
# housing["ocean_proximity"].value_counts()
#
# show a summary of the numerical attributes
# housing.describe()
#
# Plotting a histogram for each numerical attribute
# %matplotlib inline # only in a Jupyter notebook
# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))
# plt.show()
#
# 3 implemenation options for building the test-set
# (1) naive - the problem is that the test-set is reshuffled every time a new data is added.
# train_set,test_set = split_train_set(housing, 0.2)
# (2) Use a hash function to selec the test-set. The problem is that it is assumed that new data gets appened to the end of the dataset
# housing_with_id = housing.reset_index() # adds an 'index' column
# train_set, test_set = split_train_set_by_id(housing, 0.2, "index")
# (3) Using the most stable features to build a unique identifier
# housing["id"] = housing["longitude"]*1000 + housing["latitude"]
# train_set, test_set = split_train_set_by_id(housing, 0.2, "id")
# (4) Scikit-Learn has a similar function (train_test_split)
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)
# (5) random sampling may introduce sampling bias if the major attribute is not uniformly distributed. In this case median_income
# Creating 5 buckets of median income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"]< 5, 5.0 , inplace = True)
# stratified sampling (i.e. keeping the proportions of each bucket also in the test-set)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing,housing["income_cat"]) :
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]

# removing the income_cat attribute :
for set_ in (strat_train_set , strat_test_set) :
    set_.drop("income_cat",axis=1,inplace=True)

# exploring the data ...
# create a copy of the training-set
housing = strat_train_set.copy()
# visualize the data - scatter plot (longitude/latitude)
# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
# the radius of each circule represent the district's population (option s), the color represents the price (option c)
# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
#             s=housing["population"]/100,label="population",figsize=(10,7),
#             c="median_house_value",cmap=plt.get_cmap("jet"),
#             colorbar=True
#             )
# plt.legend()

# Computing standard correlation coefficients between every pair of attributes
corr_matrix = housing.corr()
# The correlation with the median house value
corr_matrix["median_house_value"].sort_values(ascending = False)
# Panda's scatter_matrix : plot every numerical attribute against every other numerical attribute

# from pandas.tools.plotting import scatter_matrix
# attributes = ["median_house_value","median_income","total_rooms",
#              "housing_median_age"]
# scatter_matrix(housing[attributes],figsize=(12,8))
# focus on the correlation with median_income
# housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.3)
# attributes combinations
#housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
#housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
#housing["population_per_room"] = housing["population"]/housing["households"]

# Prepare the data for machine learning algorithms
# 1. clean dataset (note that drop() creates a copy of the data and does not affect strat_train_set)
# 2. separate the predictors from the labels
housing = strat_train_set.drop("median_house_value" , axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Missing features -- three option
# option 1: get rid of the corresponding training examples : housing.dropna(subset=["total_bedrooms"])
# option 2: get rid of the whole attribute : housing.drop("total_bedroos",axis=1
# option 3: Set the values to some values (zero, the mean, the median, etc.):
# housing["total_bedrooms"].fillna(median,inplace=True)

# Sci-kit solution for option (3) :
from sklearn.preprocessing import Imputer
# Create an instance, specifying you want to replace each attribute's missing values with the median of that attribute
# imputer = Imputer(strategy = "median")
# The median can be computed only on numerical attributes -- we need to create a copy of the data w/o
# the text attribute 'ocean_proximity'
housing_num = housing.drop("ocean_proximity",axis=1)
# fit the imputer instance to the training data
# imputer.fit(housing_num)
# the median values of each attributes are stored in imputer.statistics_
# transform the training set by replacing the missing values by the learned medians
# X = imputer.transform(housing_num)
# convert the numpy array (X) into Pandas DataFrame
# housing_tr = pd.DataFrame(X,columns=housing_num.columns)

# converting text labels to numbers
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# The mapping is stored in encoder.classes_
# The problem with this representation is that ML algorithms will assume that two nearby algorithms are more similar
# than two distant values - the solution is to use one-hot encoding
# (note that this is a two step process : text values -> numerical values -> one-hot).

# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# The output is a sparse matrix - in order to get a dense numpy array : housing_cat_1hot.toarray()
# One step process - using LabelBinarizer
# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing_cat)
# The return value is a dense (numpy) array. In order to get a sparse matrix , you need to pass the parameter
# 'spare_output = True' to the LabelBinarizer consructor
# Custom trasformation
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix , bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__ (self, add_bedrooms_per_room = True) :
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit (self, X, y = None):
        return self
    def transform (self, X, y = None):
        rooms_per_household = X[:,rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room :
            bedrooms_per_room = X[:,bedrooms_ix] /X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,
                         bedrooms_per_room]
        else :
            return np.c_[X,rooms_per_household,population_per_household]

attr_adder = CombinedAttributeAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Tranformation pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline
# (1) Replacing each attribute's missing values with the median of that attribute
# (2) Add the combined attributes
# (3) Feature scaling - standartization (subtract by the mean and divide by the variance)

# new_pipeline = Pipeline([
#        ('imputer',Imputer(strategy="median")),
#        ('attribs_addr',CombinedAttributeAdder()),
#        ('std_scaler',StandardScaler()),
#    ])

# housing_num_tr = num_pipeline.fit_transform(housing_num)

# Custom transformation to extract numerical columns into Numpy array
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__ (self, attribute_names) :
        self.attribute_names = attribute_names
    def fit (self, X , y=None) :
        return self
    def transform (self, X, y=None):
        return X[self.attribute_names].values

# The problem with the LabelBinarizer is that the encoding depends on the values presented in the training examples
# if a certain categorial field (in the test-set , validation set) does not contain all the possible values,
# then the encoding will be different

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
#       enc = LabelBinarizer(sparse_output=self.sparse_output)
#       return enc.fit_transform(X)
        return X

# Modified pipeline - add the conversion from Pandas DataFrame
num_attribs = list(housing_num)
#cat_attribs = ["ocean_proximity"]
cat_attribs = ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","ISLAND"]

num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer',Imputer(strategy="median")),
        ('attribs_addr',CombinedAttributeAdder()),
        ('std_scaler',StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector',DataFrameSelector(cat_attribs)),
        ('label_binarizer',CustomLabelBinarizer()),
    ])

# Combining the two pipelines :
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list = [
        ("num_pipeline",num_pipeline),
        ("cat_pipeline",cat_pipeline),
    ])

# Running the whole pipeline

housing_new = housing.copy()
one_hot_list = ['<1H OCEAN','INLAND','NEAR OCEAN','NEAR BAY','ISLAND']
for i in one_hot_list :
    housing_new[i] = (housing_new["ocean_proximity"] == i)*1

housing_prepared = full_pipeline.fit_transform(housing_new)

from sklearn.linear_model import LinearRegression
from sklearn.metrics      import mean_squared_error
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor

def optimize (opt,labels,training_data) :
    if  (opt == 'lin_reg') :
        # Training Linear Regression model

        lin_reg = LinearRegression()
        lin_reg.fit(training_data,labels)
        return lin_reg
    elif (opt == 'tree_reg') :
        # Training a DecisionTreeRegressor

        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(training_data, labels)
        return tree_reg
    elif (opt == 'rand_forest'):
        # RandomForestRegressor

        rand_reg = RandomForestRegressor()
        rand_reg.fit(training_data,labels)
        return rand_reg
    else :
        return None

def opt_perf (obj, labels, training_data) :
    predictions = obj.predict(training_data)
    rmse = np.sqrt(mean_squared_error(labels,predictions))
    return rmse

# Test Set
x_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

x_test_new = x_test.copy()
one_hot_list = ['<1H OCEAN','INLAND','NEAR OCEAN','NEAR BAY','ISLAND']
for i in one_hot_list :
    x_test_new[i] = (x_test_new["ocean_proximity"] == i)*1

x_test_prepared = full_pipeline.transform(x_test_new)

# Linear Regression
obj = optimize('lin_reg',housing_labels,housing_prepared)
print ('Linear Regression: Training set', opt_perf(obj,housing_labels,housing_prepared),
                              'Test set', opt_perf(obj,y_test,x_test_prepared))

# Testing ...
# some_data = housing_new.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.fit_transform(some_data)


# Training a DecisionTreeRegressor
obj = optimize('tree_reg',housing_labels,housing_prepared)
print ('DesicionTreeRegressor: Training set', opt_perf(obj,housing_labels,housing_prepared),
                                  'Test set', opt_perf(obj,y_test,x_test_prepared))

obj = optimize('rand_forest',housing_labels,housing_prepared)
print ('RandomForestRegressor: Training set', opt_perf(obj,housing_labels,housing_prepared),
                                  'Test set', opt_perf(obj,y_test,x_test_prepared))


# 0 rmse , means over-fitting
