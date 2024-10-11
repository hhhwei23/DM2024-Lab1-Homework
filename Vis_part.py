import pandas as pd
categories = ['not nostalgia', 'nostalgia']

dataSet = pd.read_csv("Nostalgic_Sentiment_Analysis_of_YouTube_Comments_Data.csv")

print(dataSet.head()) # print top 5 lines

import pandas as pd

dataSet['sent_num'] = dataSet['sentiment'].apply(lambda x: 1 if x == 'nostalgia' else 0)

import nltk
nltk.download('punkt_tab')
X = dataSet
X['unigrams'] = X['comment'].apply(lambda x: dmh.tokenize_text(x))

X[0:5]["unigrams"]

from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()
X_vec = count_vec.fit_transform(X["comment"])
print(X_vec[0:2])

count_vec.get_feature_names_out()[1598]

analyze = count_vec.build_analyzer()

analyze(X["comment"][0])

# get only head(20)
plot_x = ["term_"+str(i) for i in count_vec.get_feature_names_out()[0:20]]

plot_x

# top 20 documents
plot_y = ["doc_"+ str(i) for i in list(X.index)[0:20]]

plot_y

plot_z = X_vec[0:20, 0:20].toarray() #X_counts[how many documents, how many terms]

plot_z

import seaborn as sns

df_todraw = pd.DataFrame(plot_z, columns = plot_x, index = plot_y)
plt.subplots(figsize=(9, 7))
ax = sns.heatmap(df_todraw,
                 cmap="PuRd",
                 vmin=0, vmax=1, annot=True)

term_frequencies = []
for j in range(0,X_vec.shape[1]):
    term_frequencies.append(sum(X_vec[:,j].toarray()))

print(X_vec)

term_frequencies = np.asarray(X_vec.sum(axis=0))[0]

term_frequencies[0]

plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vec.get_feature_names_out()[:300], 
            y=term_frequencies[:300])
g.set_xticklabels(count_vec.get_feature_names_out()[:300], rotation = 90);

import pandas as pd
import plotly.express as px

term_frequencies = np.asarray(X_vec.sum(axis=0))[0]

data = {
    'Terms':count_vec.get_feature_names_out(),
    'Freq':term_frequencies
}

df = pd.DataFrame(data)

df_filtered = df[df['Freq'] > 100]

fig = px.bar(df_filtered,
             x='Terms',
             y='Freq',
             title='Bar Chart plotly')

fig.show()

# Answer here

df_sorted = df.sort_values(by='Freq', ascending=False)

top = df_sorted.head(50)

fig = px.bar(top,
             x='Terms',
             y='Freq',
             title='Bar Chart plotly')

fig.show()

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

categories = X['sentiment'].unique()
category_dfs = {} # --> to store Dataframes for each category

for category in categories:
    category_dfs[category] = X[X['sentiment'] == category].copy()

print(len(category_dfs))

category_dfs

def create_term_document_df(df):
    count_vect = CountVectorizer()  # Initialize the CountVectorizer
    X_counts = count_vect.fit_transform(df['comment'])  # Transform the text data into word counts
    
    # Get the unique words (vocabulary) from the vectorizer
    words = count_vect.get_feature_names_out()
    
    # Create a DataFrame where rows are documents and columns are words
    term_document_df = pd.DataFrame(X_counts.toarray(), columns=words)
    
    return term_document_df

term_doc_dfs = {}

for category in categories:
    term_doc_dfs[category] = create_term_document_df(category_dfs[category])

print(len(term_doc_dfs))

term_doc_dfs

from PAMI.extras.DF2DB import DenseFormatDF as db

for category in term_doc_dfs:
    category_safe = category.replace(' ', '_')

    print(category_safe)

    obj = db.DenseFormatDF(term_doc_dfs[category])

    obj.convert2TransactionalDatabase(f'td_freq_db_{category_safe}.csv',  '>=', 1)

from PAMI.extras.dbStats import TransactionalDatabase as tds

obj = tds.TransactionalDatabase('td_freq_db_nostalgia.csv')
obj.run()
obj.printStats()
obj.plotGraphs()

from PAMI.frequentPattern.basic import FPGrowth as alg
minSup=9
obj1 = alg.FPGrowth(iFile='td_freq_db_nostalgia.csv', minSup=minSup)
obj1.mine()
frequentPatternsDF_nostalgia= obj1.getPatternsAsDataFrame()
print('Total No of patterns: ' + str(len(frequentPatternsDF_nostalgia)))

print('Runtime: ' + str(obj1.getRuntime()))

import pandas as pd

#We group together all of the dataframes related to our found patterns
dfs = [frequentPatternsDF_nostalgia, frequentPatternsDF_not_nostalgia]


# Identify patterns that appear in more than one category
# Count how many times each pattern appears across all dataframes
pattern_counts = {}
for df in dfs:
    for pattern in df['Patterns']:
        if pattern not in pattern_counts:
            pattern_counts[pattern] = 1
        else:
            pattern_counts[pattern] += 1

# Filter out patterns that appear in more than one dataframe
unique_patterns = {pattern for pattern, count in pattern_counts.items() if count == 1}
# Calculate the total number of patterns across all categories
total_patterns_count = sum(len(df) for df in dfs)
# Calculate how many patterns were discarded
discarded_patterns_count = total_patterns_count - len(unique_patterns)

# For each category, filter the patterns to keep only the unique ones
filtered_dfs = []
for df in dfs:
    filtered_df = df[df['Patterns'].isin(unique_patterns)]
    filtered_dfs.append(filtered_df)

# Merge the filtered dataframes into a final dataframe
final_pattern_df = pd.concat(filtered_dfs, ignore_index=True)

# Sort by support
final_pattern_df = final_pattern_df.sort_values(by='Support', ascending=False)

# Display the final result
print(final_pattern_df)
# Print the number of discarded patterns
print(f"Number of patterns discarded: {discarded_patterns_count}")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Convert 'text' column into term-document matrix using CountVectorizer
count_vect = CountVectorizer()
X_tdm = count_vect.fit_transform(X['comment'])  # X['text'] contains your text data
terms = count_vect.get_feature_names_out()  # Original terms in the vocabulary

# Tokenize the sentences into sets of unique words
X['tokenized_text'] = X['comment'].str.split().apply(set)

# Initialize the pattern matrix
pattern_matrix = pd.DataFrame(0, index=X.index, columns=final_pattern_df['Patterns'])

# Iterate over each pattern and check if all words in the pattern are present in the tokenized sentence
for pattern in final_pattern_df['Patterns']:
    pattern_words = set(pattern.split())  # Tokenize pattern into words
    pattern_matrix[pattern] = X['tokenized_text'].apply(lambda x: 1 if pattern_words.issubset(x) else 0)

# Convert the term-document matrix to a DataFrame for easy merging
tdm_df = pd.DataFrame(X_tdm.toarray(), columns=terms, index=X.index)

# Concatenate the original TDM and the pattern matrix to augment the features
augmented_df = pd.concat([tdm_df, pattern_matrix], axis=1)

augmented_df

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

#This might take a couple of minutes to execute
# Apply PCA, t-SNE, and UMAP to the data
X_pca_tdm = PCA(n_components=2).fit_transform(tdm_df.values)
X_tsne_tdm = TSNE(n_components=2).fit_transform(tdm_df.values)
X_umap_tdm = umap.UMAP(n_components=2).fit_transform(tdm_df.values)

# Plot the results in subplots
col = ['coral', 'blue', 'black', 'orange']
categories = X['sentiment'].unique() 

fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Create 3 subplots for PCA, t-SNE, and UMAP
fig.suptitle('PCA, t-SNE, and UMAP Comparison')

# Define a function to create a scatter plot for each method
def plot_scatter(ax, X_reduced, title):
    for c, category in zip(col, categories):
        xs = X_reduced[X['sentiment'] == category].T[0]
        ys = X_reduced[X['sentiment'] == category].T[1]
        ax.scatter(xs, ys, c=c, marker='o', label=category)
    
    ax.grid(color='gray', linestyle=':', linewidth=2, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')

# Step 4: Create scatter plots for PCA, t-SNE, and UMAP
plot_scatter(axes[0], X_pca_tdm, 'PCA')
plot_scatter(axes[1], X_tsne_tdm, 't-SNE')
plot_scatter(axes[2], X_umap_tdm, 'UMAP')

plt.show()
