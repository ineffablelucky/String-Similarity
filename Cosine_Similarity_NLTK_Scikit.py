# imports
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()

# NLTK Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# NLTK tokenization
def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# two test strings
# string_1 = "The sun in the sky is bright"
# string_2 = "The sun is bright"
string_1 = input("Enter string one : ")
string_2 = input("Enter string two : ")

# this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs_matrix = tfidf.fit_transform([string_1, string_2])


result_array = cosine_similarity(tfs_matrix[0:1], tfs_matrix[1:])  # the first element of tfs_matrix is matched with other elements
print("cosine similarity score of two strings is (range from 0 to 1):  " + str(result_array[0][0]))