import pandas as pd

# Creating data
# There are two core objects in pandas: the DataFrame and the Series.
# A DataFrame is a table. It contains an array of individual entries, each of which has a certain value. Each entry corresponds
# with a row (or record) and a column. For example, consider the following simple DataFrame:

pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

# two columns 'Yes' and 'No' : The column 'Yes' contains the numbers 50,21; the column 'No' contains the numbers 131,2
# two rows - 0 , 1 - Row '0' : 50,131 ; Row '1' : 21,2

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
# Similar idea , the values are strings

# We are using the pd.DataFrame constructor to generate these DataFrame objects. The syntax for declaring a new one is a dictionary
# whose keys are the column names (Bob and Sue in this example), and whose values are a list of entries. This is the standard way of
# constructing a new DataFrame, and the one you are likliest to encounter.


# The dictionary-list constructor assigns values to the column labels, but just uses an ascending count from 0 (0, 1, 2, 3, ...) for
# the row labels. Sometimes this is OK, but oftentimes we will want to assign these labels ourselves.
# The list of row labels used in a DataFrame is known as an Index. We can assign values to it by using an index parameter in our
# constructor:
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])


# A Series, by contrast, is a sequence of data values. If a DataFrame is a table, a Series is a list. And in fact you can create one
# with nothing more than a list:
pd.Series([1, 2, 3, 4, 5])

# A Series is, in essence, a single column of a DataFrame. So you can assign column values to the Series the same way as before,
# using an index parameter. However, a Series do not have a column name, it only has one overall name:
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

# Reading common file formats
# a CSV file is a table of values separated by commas. Hence the name: "comma-seperated values", or CSV.
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
# We can use the shape attribute to check how large the resulting DataFrame is:
wine_reviews.shape
# We can examine the contents of the resultant DataFrame using the head command, which grabs the first five rows:
wine_reviews.head()

# you can see in this dataset that the csv file has an in-built index, which pandas did not pick up on automatically. To make pandas
# use that column for the index (instead of creating a new one from scratch), we may specify and use an index_col:
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# SQL files
# Connecting to a SQL database requires a lot more thought than reading from an Excel file. For one, you need to create a connector,
# something that will handle siphoning data from the database.
#
# pandas won't do this for you automatically because there are many, many different types of SQL databases out there, each with its own
# connector. So for a SQLite database (the only kind supported on Kaggle), you would need to first do the following (using the sqlite3
# library that comes with Python):
import sqlite3
conn = sqlite3.connect("../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite")

# here is all the SQL you have to know to get the data out of SQLite and into pandas
fires = pd.read_sql_query("SELECT * FROM fires", conn)
# Every SQL statement begins with SELECT. The asterisk (*) is a wildcard character, meaning "everything", and FROM fires tells the
# database we want only the data from the fires table specifically.

# Writing common file formats
# Writing data to a file is usually easier than reading it out of one, because pandas handles the nuisance of conversions for you.

# The opposite of read_csv, which reads our data, is to_csv, which writes it.
wine_reviews.head().to_csv("wine_reviews.csv")

# To write an Excel file back you need to_excel
# And finally, to output to a SQL database, supply the name of the table in the database we want to throw the data into, and a
# connector:
conn = sqlite3.connect("fires.sqlite")
fires.head(10).to_sql("fires", conn)

#
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
# In Python we can access the property of an object by accessing it as an attribute. A book object, for example, might have a title
# property, which we can access by calling book.title. Columns in a pandas DataFrame work in much the same way.
# Hence to access the country property of our reviews we can use:
reviews.country

# If we have a dict object in Python, we can access its values using the indexing ([]) operator. Again, we can do the same with
# pandas DataFrame columns. It "just works":
reviews['country']
# to drill down to a single specific value, we need only use the indexing operator [] once more:
reviews['country'][0]

# pandas indexing works in one of two paradigms. The first is index-based selection: selecting data based on its numerical position
# in the data. iloc follows this paradigm. To select the first row of data in this DataFrame, we may use the following:
reviews.iloc[0]
# Both loc and iloc are row-first, column-second. This is the opposite of what we do in native Python, which is column-first,
# row-second. This means that it's marginally easier to retrieve rows, and marginally harder to get retrieve columns. To get a column
# with iloc, we can do the following:
reviews.iloc[:, 0]

# to select just the second and third entries, we would do:
reviews.iloc[1:3, 0]
# It's also possible to pass a list:
reviews.iloc[[0, 1, 2], 0]

# Finally, it's worth knowing that negative numbers can be used in selection. This will start counting forwards from the end of the
# values. So for example here are the last five elements of the dataset.
reviews.iloc[-5:]

# The second paradigm for attribute selection is the one followed by the loc operator: label-based selection. In this paradigm it's
# the data index value, not its position, which matters.For example, to get the first entry in reviews, we would now do the
# following:
reviews.loc[0, 'country']

# When choosing or transitioning between loc and iloc, there is one "gotcha" worth keeping in mind, which is that the two methods use
# slightly different indexing schemes. iloc uses the Python stdlib indexing scheme, where the first element of the range is included
# and the last one excluded. So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. So 0:10 will select
# entries 0,...,10.

# Why the change? Remember that loc can index any stdlib type: strings, for example. If we have a DataFrame with index values
# Apples, ..., Potatoes, ..., and we want to select "all the alphabetical fruit choices between Apples and Potatoes", then it's a
# heck of a lot more convenient to index df.loc['Apples':'Potatoes'] than it is to index something like
# df.loc['Apples', 'Potatoet] (t coming after s in the alphabet).


# Label-based selection derives its power from the labels in the index. Critically, the index we use is not immutable. We can
# manipulate the index in any way we see fit. The set_index method can be used to do the job. Here is what happens when we set_index
# to the title field:
reviews.set_index("title")

reviews.country == 'Italy'
# This operation produced a Series of True/False booleans based on the country of each record. This result can then be used inside
# of loc to select the relevant data:
reviews.loc[reviews.country == 'Italy']
# Multiple conditions :
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
# select wines only from Italy or France:
reviews.loc[reviews.country.isin(['Italy', 'France'])]

# filter out wines lacking a price tag in the dataset
reviews.loc[reviews.price.notnull()]

# Assigning data
# Going the other way, assigning data to a DataFrame is easy. You can assign either a constant value:
reviews['critic'] = 'everyone'

# Or with an iterable of values:
reviews['index_backwards'] = range(len(reviews), 0, -1)

reviews.points.describe()
# This method generates a high-level summary of the attributes of the given column. It is type-aware, meaning that its output changes
# based on the dtype of the input.
# mean
reviews.points.mean()

# To see a list of unique values:
reviews.taster_name.unique()

# To see a list of unique values and how often they occur in the dataset, we can use the value_counts method:
reviews.taster_name.value_counts()

# suppose that we wanted to re-mean the scores the wines recieved to 0. We can do this as follows:
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

# The function you pass to map should expect a single value from the Series (a point value, in the above example), and return a
# transformed version of that value. map returns a new Series where all the values have been transformed by your function.

# DataFrame.apply is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')

# If we had called reviews.apply with axis='index', then instead of passing a function to transform each row, we would need to
# give a function to transform each column.

# Note that Series.map and DataFrame.apply return new, transformed Series and DataFrames, respectively. They don't modify the
# original data they're called on.

# pandas provides many common mapping operations as built-ins. For example, here's a faster way of remeaning our points column:
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean

# pandas will also understand what to do if we perform these operations between Series of equal length
reviews.country + " - " + reviews.region_1

# Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column. However, often we want to
# group our data, and then do something specific to the group the data is in. To do this, we can use the groupby operation.
# For example, one function we've been using heavily thus far is the value_counts function. We can replicate what value_counts
# does using groupby by doing the following:
reviews.groupby('points').points.count()

# to get the cheapest wine in each point value category:
reviews.groupby('points').price.min()

# here's one way of selecting the name of the first wine reviewed from each winery in the dataset:
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

# For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the
# best wine by country and province:
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.argmax()])

# agg lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple
# statistical summary of the dataset as follows:
reviews.groupby(['country']).price.agg([len, min, max])

# In all of the examples we've seen thus far we've been working with DataFrame or Series objects with a single-label index. groupby is
# slightly different in the fact that, depending on the operation we run, it will sometimes result in what is called a multi-index.

# converting back to a regular index :
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed.reset_index()

# Sorting
# To get data in the order want it in we can sort it ourselves. The sort_values method is handy for this.
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')

# sort_values defaults to an ascending sort, where the lowest values go first. Decending order :
countries_reviewed.sort_values(by='len', ascending=False)

# To sort by index values, use the companion method sort_index :
countries_reviewed.sort_index()
# you can sort by more than one column at a time:
countries_reviewed.sort_values(by=['country', 'len'])

# Data types
# The data type for a column in a DataFrame or a Series is known as the dtype.

# It's possible to convert a column of one type into another wherever such a conversion makes sense by using the astype function
reviews.points.astype('float64')

# Entries missing values are given the value NaN, short for "Not a Number". For technical reasons these NaN values are always of
# the float64 dtype.

# pandas provides some methods specific to missing data. To select NaN entreis you can use pd.isnull (or its companion pd.notnull)

# Replacing missing values is a common operation. pandas provides a really handy method for this problem: fillna. fillna provides
# a few different strategies for mitigating such data. For example, we can simply replace each NaN with an "Unknown":
reviews.region_2.fillna("Unknown")

# Or we could fill each missing value with the first non-null value that appears sometime after the given record in the database.
# This is known as the backfill strategy:

# Suppose that since this dataset was published, reviewer Kerin O'Keefe has changed her Twitter handle from @kerinokeefe to @kerino.
# One way to reflect this in the dataset is using the replace method:
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")

# Renaming
# Oftentimes data will come to us with column names, index names, or other naming conventions that we are not satisfied with. In
# that case, we may use pandas renaming utility functions to change the names of the offending entries to something better.

# To change the points column in our dataset to score, we would do:
reviews.rename(columns={'points': 'score'})
# rename lets you rename index or column values by specifying a index or column keyword parameter, respectively. It supports a
# variety of input formats, but I usually find a Python dict to be the most convenient one. Here is an example using it to rename
# some elements on the index.
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
# You'll probably rename columns very often, but rename index values very rarely. For that, set_index is usually more convenient.

# Both the row index and the column index can have their own name attribute. The complimentary rename_axis method may be used to
# change these names. For example:
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

# Combining
# When performing operations on a dataset we will sometimes need to combine different DataFrame and/or Series in non-trivial ways.
# pandas has three core methods for doing this. In order of increasing complexity, these are concat, join, and merge. Most of what
# merge can do can also be done more simply with join
# The simplest combining method is concat. This function works just like the list.concat method in core Python: given a list of
# elements, it will smush those elements together along an axis.
# This is useful when we have data in different DataFrame or Series objects but having the same fields (columns)
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])
# join lets you combine different DataFrame objects which have an index in common. For example, to pull down videos that happened
# to be trending on the same day in both Canada and the UK, we could do the following:
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK')

# The lsuffix and rsuffix parameters are necessary here because the data has the same column names in both British and Canadian
# datasets. If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.
