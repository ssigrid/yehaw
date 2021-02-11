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

def read_file(file):

    """ Function that opens the file, reads it, and writes the data in it
    into a list where every wiki article is its own item in the list
    """

    openfile = open(file, "r", encoding='utf-8')
    print("Writing the file into a list...")
    readfile = openfile.read()
    filelist = readfile.split("</article>")
    filelist = filelist[:-1]                # Deletes an empty line from the end of the list
    openfile.close()
    print("Success!")
    print()
    
    return filelist     # returns the data that's been split into a list
    
def stem_file(file):

    """ Function that reads every line of the file, puts it through a stemmer,
    and writes the stemmed text into a new list
    """

    stemlist = []
    print("Analyzing words...")
    for line in file:
        line = stemmer(line)
        stemlist.append(line)
    print("Success!")
    print()
    
    return stemlist     # returns the list with the stemmed sentences

def get_name(file):
    
    """ Function that searches the articles for the article tag,
    substitutes the tag with the actual title of the article and a newline character,
    then splits the article into a list based on newline characters,
    and finally writes the first line of the split article into a new list,
    creating a list of titles corresponding to the articles
    """
    
    namelist = []
    
    print("Extracting article titles...")
    for i in wikidoc:
        find_tag = re.sub(r'\n?<article name="(([A-Za-z \(\)0-9])+)">', r'\1\n', i)
        find_title = re.split(r'\n', find_tag)
        namelist.append(find_title[0])
    print("Success!")
    print()
    
    return namelist     # returns the list of article titles

def search_file(query_string):

    """ Function that searches the query from the (non-stemmed) articles
    by using the cosine similarity of the query and the articles,
    then sorts the hits based on relevance, and prints the titles of
    the 15 most relevant articles
    """
    
    # Vectorize query string
    query_vec = gv1.transform([ query_string ]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, g_matrix)

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    
    # Output result
    if (len(ranked_scores_and_doc_ids)) <= 15:
        print("Your query '{:s}' matches the following documents:".format(query_string))
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, wikinames[doc_idx]))
    else:
        print("Your query '{:s}' best matches the following 15 documents:".format(query_string))
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 14:
                print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, wikinames[doc_idx]))
    print()

def search_file_stem(query_string):

    """ Function that runs the query through the stemmer, searches that query
    from the stemmed articles by using the cosine similarity of the query and the articles,
    then sorts the hits based on relevance, and prints the titles of
    the 15 most relevant articles
    """
    
    query_stem = stemmer(query_string)
    
    # Vectorize query string
    query_vec = gv2.transform([ query_stem ]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, g_matrix_stem)

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    
    # Output result
    if (len(ranked_scores_and_doc_ids)) <= 15:
        print("Your query '{:s}' matches the following documents:".format(query_string))
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, wikinames[doc_idx]))
    else:
        print("Your query '{:s}' best matches the following 15 documents:".format(query_string))
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 14:
                print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, wikinames[doc_idx]))
    print()

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

wiki = "wiki100.txt"
wikidoc = read_file(wiki)
wikinames = get_name(wikidoc)
wikistem = stem_file(wikidoc)

#print(wikidoc[-1])
#print(wikinames[-1])
#print(wikistem[-1])

gv1 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix = gv1.fit_transform(wikidoc).T.tocsr()
gv2 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix_stem = gv2.fit_transform(wikistem).T.tocsr()

#print("There are", len(wikidoc), "books in the collection:", wikinames)
#print("Number of terms in vocabulary:", len(gv.get_feature_names()))

def main():
    
    """ Main program that asks the user for a query and if they want to
    search based on the stems of the words,
    then directs the query to a function based on the user's answer
    """
    
    loop = True
    while loop == True:
        print("Input query or an empty string to quit: ")
        query = input()
        if query == "":
            print("Goodbye!")
            loop = False
        else:
            if re.search(r'^".*"$', query): ## tänne normi jos yks sana exact 
                print()
                try:
                    search_file(query)
                except IndexError:
                    print("IndexError: Your query didn't match any documents.")
               # if re.search(r'\*', query):
                
            else: ## normi stemmi
                print()
                try:
                    search_file_stem(query)
                except IndexError:
                    print("IndexError: Your query didn't match any documents.")
            

main()
