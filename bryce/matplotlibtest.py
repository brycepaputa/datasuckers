import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer

nrows=50000
# This is a list of the features that we wil be importing.  Feel free to lengthen or
# shorten the list, as needed.
feature_columns = ['id','textbookuse','interest','grade','helpcount','nothelpcount',
                   'online','profgender','profhotness','helpfulness','clarity','easiness','quality']
train_data_features = pd.read_csv("./xtrain_clean.csv", nrows=nrows, usecols=feature_columns)

# Some of the columns contain NaN placeholders.  We won't be able to very much while they are there.
# Fortunately, Pandas has a fillna() command that replaces them with a specified value.  In this case,
# we chose the mean.
train_data_features.fillna(train_data_features.mean(), inplace=True)
col_names = train_data_features.columns.tolist()

# First, import a list of stop words that will be ignored by the CountVectorizer
stop_words = './stop_words.txt'
with open(stop_words, 'r') as f:
    words = f.read()
    words = words.split('\n')[:-1]
# Now, use pandas to import the comment data
train_data_comments = pd.read_csv("./xtrain_clean.csv", nrows=nrows, usecols=['id', 'comments'])

# Create a CountVectorizer object and use the fit_transform method to learn the document-term matrix.
count_vec = CountVectorizer(min_df=120, ngram_range=(1,2), stop_words=words)
doc_word_matrix = count_vec.fit_transform(train_data_comments['comments'].fillna(''))

# DataFrame objects can be created from dictionaries.  We start by initializing a dictionary
# that has just the ids corresponding to comments.
freq_dict = {'id':train_data_comments.id}
count = 0

# The count_vec object stores the words from the text as a list.  Word i corresponds to index i.
for word in count_vec.get_feature_names():
    # Pull a column from the document-term matrix and assign it to its corresponding word
    # in the dictionary.
    freq_dict[word] = doc_word_matrix[:,count].toarray().T[0]  # I admit it, this looks hacky and gross.
    count+=1
    
# Create a new DataFrame object with the dictionary we created.  This will be much easier to search
# and analyze later.
doc_word_df = pd.DataFrame(data=freq_dict)

# Find duplicate rows.  Based on my observations, a duplicate of a row always immediately follows the row of which it 
# is a copy.  Thus, we will go row by row and see if any two subsequent ids are the same.  If so, we will add the 
# index of the id to a list for future removal.
duplicates = []
for i in range(len(doc_word_df)-1):
    if doc_word_df['id'][i] == doc_word_df['id'][i+1]:
        duplicates.append(i)
        
# We will start by considering the professor's gender.  You should experiment with other columns.
text_analysis_columns = ['id', 'profgender']
columns_df = train_data_features[text_analysis_columns]

# We create some temporary dataframes for dropping duplicate rows.
# If you don't see why I do this, don't worry too much about it for now.
# It is mostly for convenience in working with the notebook.
temp_columns_df = columns_df.drop(duplicates)
temp_doc_word_df = doc_word_df.drop(duplicates)

# Create a new DataFrame by merging the document-term matrix with the columns specified in
# the text_analysis_columns list.
experiment_df = temp_doc_word_df.merge(temp_columns_df, on='id')

# We are done with these temporary DataFrames, so set them to None to avoid using up too much memory.
temp_doc_word_df = None
temp_columns_df = None

# This calculates the correlation matrix.
feats_corr_matrix = np.corrcoef(train_data_features.drop(['id'], axis=1), rowvar=0)

# This code builds the plot.  Most of the code is to make the image pretty.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Feature Correlation Matrix')
ax.imshow(feats_corr_matrix, interpolation='none', cmap='gray')
ax.annotate('Clarity, Helpfulness', xy=(9, 8), xytext=(1, 6), color='red',
            arrowprops=dict(facecolor='red', shrink=0.05))
            
            # Create a scatter plot of clarity vs helpfulness scores.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Scatter plot of clarity and helpfulness scores")
ax.scatter(train_data_features['clarity'], train_data_features['helpfulness'])

# This creates a histogram. Is this more informative than the scatter plot?
fig = plt.figure()
fig.set_size_inches(16, 5)
ax.set_title("Correlation clarity and helpfulness")
ax = fig.add_subplot(121, projection='3d')

# These values give the range of values the features may take. In
# this example, clarity is on the x axis and helpfulness is on the
# y axis.
xx = np.arange(0,6,1)
yy = np.arange(0,6,1)

# Dealing with 3d graphs is a pain.  What follows is a framework that you
# can use to create your own.
xpos, ypos = np.meshgrid(xx,yy)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(len(ypos))
dx = 0.5*np.ones_like(zpos)
dy = dx.copy()

# The zz values give how tall each bar in our graph is.
zz = np.zeros((len(xx), len(yy)))
for k in range(len(train_data_features['clarity'])):
    # Increment zz[i,j] when clarity = i and helpfulness = j
    zz[train_data_features['clarity'][k], train_data_features['helpfulness'][k]] += 1

# Save the zz matrix for later display as an image or heatmap.
im_zz = zz.copy()

# More 3d plot stuff.
dz = zz.flatten(order='F')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
ax.set_title('3d-Histogram')

# Plot the im_zz matrix as a heatmap.
ax = fig.add_subplot(122)
ax.imshow(im_zz, interpolation='none', origin='lower')
ax.set_title('Heatmap')
plt.show()
