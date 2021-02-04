import nltk
nltk.download('gutenberg')
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


documents = ["This is a silly silly silly example",
             "A better example",
             "Nothing to see here nor here nor here",
             "This is a great example and a long example too"]



cv = CountVectorizer(lowercase=True, binary=True)
binary_dense_matrix = cv.fit_transform(documents).T.todense()

print("Term-document matrix:\n")
print(binary_dense_matrix)

cv = CountVectorizer(lowercase=True)
dense_matrix = cv.fit_transform(documents).T.todense()

print("Term-document matrix:\n")
print(dense_matrix)

for (row, term) in enumerate(cv.get_feature_names()):
    print("Row", row, "is the vector for term:", term)

t2i = cv.vocabulary_  # shorter notation: t2i = term-to-index
print("Query: example")
print(dense_matrix[t2i["example"]])

hits_list = np.array(dense_matrix[t2i["example"]])[0]

for i, nhits in enumerate(hits_list):
    print("Example occurs", nhits, "time(s) in document:", documents[i])

print("Query: better example")
print("Hits of better:        ", dense_matrix[t2i["better"]])
print("Hits of example:       ", dense_matrix[t2i["example"]])
print("Hits of better example:", dense_matrix[t2i["better"]] + dense_matrix[t2i["example"]])

print("Query: silly example")
print("Hits of silly:        ", dense_matrix[t2i["silly"]])
print("Hits of example:      ", dense_matrix[t2i["example"]])
print("Hits of silly example:", dense_matrix[t2i["silly"]] + dense_matrix[t2i["example"]])

# We need the np.array(...)[0] code here to convert the matrix to an ordinary list:
hits_list = np.array(dense_matrix[t2i["silly"]] + dense_matrix[t2i["example"]])[0]
print("Hits:", hits_list)

nhits_and_doc_ids = [ (nhits, i) for i, nhits in enumerate(hits_list) if nhits > 0 ]
print("List of tuples (nhits, doc_idx) where nhits > 0:", nhits_and_doc_ids)

ranked_nhits_and_doc_ids = sorted(nhits_and_doc_ids, reverse=True)
print("Ranked (nhits, doc_idx) tuples:", ranked_nhits_and_doc_ids)

print("\nMatched the following documents, ranked highest relevance first:")
for nhits, i in ranked_nhits_and_doc_ids:
    print("Score of 'silly example' is", nhits, "in document:", documents[i])

# Parameters with which TfidfVectorizer does same thing as CountVectorizer
tfv1 = TfidfVectorizer(lowercase=True, sublinear_tf=False, use_idf=False, norm=None)
tf_matrix1 = tfv1.fit_transform(documents).T.todense()

print("TfidfVectorizer:")
print(tf_matrix1)

print("\nCountVectorizer:")
print(dense_matrix)

tfv2 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=False, norm=None)
tf_matrix2 = tfv2.fit_transform(documents).T.todense()

print("TfidfVectorizer (logarithmic term frequencies):")
print(tf_matrix2)

tfv3 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm=None)
tf_matrix3 = tfv3.fit_transform(documents).T.todense()

print("TfidfVectorizer (logarithmic term frequencies and inverse document frequencies):")
print(tf_matrix3)

tfv4 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
tf_matrix4 = tfv4.fit_transform(documents).T.todense()

print("TfidfVectorizer (logarithmic term frequencies and inverse document frequencies, normalized document vectors):")
print(tf_matrix4)

print("Query: silly example")
print("Hits of silly:        ", tf_matrix4[t2i["silly"]])
print("Hits of example:      ", tf_matrix4[t2i["example"]])
print("Hits of silly example:", tf_matrix4[t2i["silly"]] + tf_matrix4[t2i["example"]])

hits_list4 = np.array(tf_matrix4[t2i["silly"]] + tf_matrix4[t2i["example"]])[0]
print("Hits:", hits_list4)

hits_and_doc_ids = [ (hits, i) for i, hits in enumerate(hits_list4) if hits > 0 ]
print("List of tuples (hits, doc_idx) where hits > 0:", hits_and_doc_ids)

ranked_hits_and_doc_ids = sorted(hits_and_doc_ids, reverse=True)
print("Ranked (hits, doc_idx) tuples:", ranked_hits_and_doc_ids)

print("\nMatched the following documents, ranked highest relevance first:")
for hits, i in ranked_hits_and_doc_ids:
    print("Score of 'silly example' is {:.4f} in document: {:s}".format(hits, documents[i]))

query_vec4 = tfv4.transform(["silly example"]).todense()
print(query_vec4)

print(query_vec4.T)

print("Tf-idf weight of 'example' on row", t2i["example"], "is:", query_vec4.T[t2i["example"]])
print("Tf-idf weight of 'silly' on row", t2i["silly"], "is: ", query_vec4.T[t2i["silly"]])

for i in range(0, 4):
    
    # Go through each column (document vector) in the index 
    doc_vector = tf_matrix4[:, i]
    
    # Compute the dot product between the query vector and the document vector
    # (Some extra stuff here to extract the number from the matrix data structure)
    score = np.array(np.dot(query_vec4, doc_vector))[0][0]
    
    print("The score of 'silly example' is {:.4f} in document: {:s}".format(score, documents[i]))

scores = np.dot(query_vec4, tf_matrix4)
print("The documents have the following cosine similarities to the query:", scores)

ranked_scores_and_doc_ids = \
    sorted([ (score, i) for i, score in enumerate(np.array(scores)[0]) if score > 0], reverse=True)

for score, i in ranked_scores_and_doc_ids:
    print("The score of 'silly example' is {:.4f} in document: {:s}".format(score, documents[i]))

tfv5 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
sparse_matrix = tfv5.fit_transform(documents).T.tocsr() # CSR: compressed sparse row format => order by terms

print("Sparse term-document matrix with tf-idf weights:")
print(sparse_matrix)

# The query vector is a horizontal vector, so in order to sort by terms, we need to use CSC
query_vec5 = tfv5.transform(["silly example"]).tocsc() # CSC: compressed sparse column format

print("Sparse one-row query matrix (horizontal vector):")
print(query_vec5)

hits = np.dot(query_vec5, sparse_matrix)

print("Matching documents and their scores:")
print(hits)

print("The matching documents are:", hits.nonzero()[1])

print("The scores of the documents are:", np.array(hits[hits.nonzero()])[0])

ranked_scores_and_doc_ids = sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)

for score, i in ranked_scores_and_doc_ids:
    print("The score of 'silly example' is {:.4f} in document: {:s}".format(score, documents[i]))


booknames = nltk.corpus.gutenberg.fileids()

bookdata = list(nltk.corpus.gutenberg.raw(name) for name in booknames)

print("There are", len(bookdata), "books in the collection:", booknames)

gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
g_matrix = gv.fit_transform(bookdata).T.tocsr()

print("Number of terms in vocabulary:", len(gv.get_feature_names()))

def search_gutenberg(query_string):

    # Vectorize query string
    query_vec = gv.transform([ query_string ]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, g_matrix)

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
               reverse=True)
    
    # Output result
    print("Your query '{:s}' matches the following documents:".format(query_string))
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, booknames[doc_idx]))
    print()

def stemmer(file):
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    porter = PorterStemmer()
    
    token_words = word_tokenize(file)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
    
search_gutenberg("alice")
search_gutenberg("alice entertained harriet")
search_gutenberg("whale hunter")
search_gutenberg("oh thy lord cometh")
search_gutenberg("which book should i read")
