from flask import Flask, render_template, request
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#Initialize Flask instance
app = Flask(__name__)

def stemmer(file):
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    porter = PorterStemmer()

    token_words = word_tokenize(file)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
    
# WIKIPEDIA ARTICLE EDITING FUNCTIONS:

def read_file(file):

    """ Function that opens the file, reads it, and writes the data in it                                      
    into a list where every wiki article is its own item in the list                                           
    """

    openfile = open(file, "r", encoding='utf-8')
    readfile = openfile.read()
    filelist = readfile.split("</article>")
    filelist = filelist[:-1]                # Deletes an empty line from the end of the list                   
    openfile.close()
    

    return filelist     # returns the data that's been split into a list

def stem_file(file):

    """ Function that reads every line of the file, puts it through a stemmer,                                 
    and writes the stemmed text into a new list                                                                
    """

    stemlist = []
    for line in file:
        line = stemmer(line)
        stemlist.append(line)

    return stemlist     # returns the list with the stemmed sentences

def get_name(file):

    """ Function that searches the articles for the article tag,                                               
    substitutes the tag with the actual title of the article and a newline character,                          
    then splits the article into a list based on newline characters,                                           
    and finally writes the first line of the split article into a new list,                                    
    creating a list of titles corresponding to the articles                                                    
    """

    namelist = []

    for i in wikidoc:
        find_tag = re.sub(r'\n?<article name="(([A-Za-z \(\)0-9])+)">', r'\1\n', i)
        find_title = re.split(r'\n', find_tag)
        namelist.append(find_title[0])


    return namelist     # returns the list of article titles
    
# RELEVANCE SEARCH FUNCTIONS:

def search_file(query):

    """ Function that searches the query from the (non-stemmed) articles
    by using the cosine similarity of the query and the articles,
    then sorts the hits based on relevance, and makes a dictionary consisting of
    the result articles' names, first 25 words, score, and rank
    """
    
    # Initialize list of results
    results = []
    
    # Vectorize query string
    query_vec = gv1.transform([ query ]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, g_matrix)

    # Rank hits
    ranked_scores_and_doc_ids = \
    sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    
    # Output result
    # Make the relevance search results into dictionary entries for name, number of hits, and rank,
    # and append them into the results list
    if (len(ranked_scores_and_doc_ids)) == 1:
        #results.append("Your query {:s} matches the following document:".format(query))
        resultsitem = {'name': "Article Title", 'text': "Your query \"{:s}\" matched one article. Here are the first 25 words of it:".format(query), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)   # we're only printing four decimals now
            resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    elif (len(ranked_scores_and_doc_ids)) <= 10:
        #results.append("Your query {:s} matches the following {:d} documents:".format(query, len(ranked_scores_and_doc_ids)))
        resultsitem = {'name': "Article Title", 'text': "Your query \"{:s}\" matched {:d} articles. Here are the first 25 words of them in order of relevance:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)
            resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    else:
        #results.append("Your query {:s} matches {:d} documents, out of which it matches the following 10 documents the best:".format(query, len(ranked_scores_and_doc_ids)))
        resultsitem = {'name': "Article Title", 'text': "Your query \"{:s}\" matched {:d} articles. Here are the first 25 words of the 10 articles your query matched the best:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 9:
                roundedscore = "{:.4f}".format(score)
                resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'score': roundedscore, 'rank': i+1}
                results.append(resultsitem)
                    
    return results      # return dictionary of results

def search_file_stem(query):

    """ Function that runs the query through the stemmer, searches that query
    from the stemmed articles by using the cosine similarity of the query and the articles,
    then sorts the hits based on relevance, and makes a dictionary consisting of
    the result articles' names, first 25 words, score, and rank
    """
    
    # Initialize list of results
    results = []
    
    query_stem = stemmer(query)
    
    # Vectorize query string
    query_vec = gv2.transform([ query_stem ]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, g_matrix_stem)

    # Rank hits
    ranked_scores_and_doc_ids = \
    sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    
    # Output result
    if (len(ranked_scores_and_doc_ids)) == 1:
        #results.append("Your query '{:s}' matches the following document:".format(query))
        resultsitem = {'name': "Article Title", 'text': "Your query '{:s}' matched one article. Here are the first 25 words of it:".format(query), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)
            resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    elif (len(ranked_scores_and_doc_ids)) <= 10:
        #results.append("Your query '{:s}' matches the following {:d} documents:".format(query, len(ranked_scores_and_doc_ids)))
        resultsitem = {'name': "Article Title", 'text': "Your query '{:s}' matched {:d} articles. Here are the first 25 words of them in order of relevance:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)
            resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    else:
        #results.append("Your query '{:s}' matches {:d} documents, out of which it matches the following 10 documents the best:".format(query, len(ranked_scores_and_doc_ids)))
        resultsitem = {'name': "Article Title", 'text': "Your query '{:s}' matched {:d} articles. Here are the first 25 words of the 10 articles your query matched the best:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 9:
                roundedscore = "{:.4f}".format(score)
                resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'score': roundedscore, 'rank': i+1}
                results.append(resultsitem)
                
    return results      # return dictionary of results

# BOOLEAN SEARCH FUNCTIONS:

def rewrite_token(t):
    d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):
    
    """ Function that rewrites the query with other functions, writes it into a list,
    replacing words that cause KeyError with a word that doesn't (but still doesn't return any hits),
    and then makes that list into a string again
    """
    
    rewritten = rewrite_query(query)
    r_list = []
    for t in rewritten.split():
        try:
            eval(t)                                         # Eval runs the string as a Python command
            r_list.append(t)
        except KeyError:                                    # I used 'not the' because the word the appears in all articles
            r_list.append('1 - td_matrix[t2i["the"]]')      # (Not how it's supposed to be done but I ran out of ideas lål)
        except SyntaxError:
            r_list.append(t)                                # Rewritten and/or/not stay the same
    r_join = " ".join(r_list)
    
    return r_join   # returns the rewritten list
    
# ARTICLE EXCERPT FUNCTION:
    
def first_25_words(article):

    """ Function that searches the article for it's article name tag, removes the tag,
    splits the article into a list with every line consisting of one line,
    creates a new list, and writes the first 25 words of the first list into the new list,
    and finally writes the new list into string
    """
    
    first_25 = re.sub(r'<article name="(([A-Za-z \(\)0-9])+)">', r'', article)
    word_list = first_25.split()
    twentyfivewords = []
    for i in range(26):
        twentyfivewords.append(word_list[i])
    twentyfivewords = ' '.join(twentyfivewords)
    twentyfivewords = twentyfivewords + "..."
    
    return twentyfivewords      # return the first 25 words of the article as string

# WIKIPEDIA ARTICLES GETTING HANDLED:

wiki = "wiki100.txt"
wikidoc = read_file(wiki)
wikinames = get_name(wikidoc)
wikistem = stem_file(wikidoc)
wikidictionary = []

for i in range(100):    # initializing the dictionary entries might be redundant, check later
    dictionarynew = {'name': wikinames[i], 'text': first_25_words(wikidoc[i])}
    wikidictionary.append(dictionarynew)
    
# VECTORS AND MATRICES INITIALIZED:

cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(wikidoc)
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_

gv1 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix = gv1.fit_transform(wikidoc).T.tocsr()
gv2 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix_stem = gv2.fit_transform(wikistem).T.tocsr()

#  PLANNING
def seabornplot(matches):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="white", context="talk")

    x = []
    y = []
    for pair in matches:
        x.append(pair[0])
        y.append(pair[1])
    
    sns.barplot(x, y)
    sns.color_palette("rocket")
    plt.xticks(rotation=45)
    plt.show()
    
    
def wordfreq(articles):
    dict_articles = []
    for article in articles: 
        dictionary = {}
        words = article.split()
        for word in words:
            if word.lower() not in dictionary:
                dictionary[word.lower()] = 1
            else:
                dictionary[word.lower()] += 1
        dict_articles.append(dictionary)
    return dict_articles


def freqhits(wordfreqs, streotype):
    matches = {}
    book_number = 0
    for dictio in wordfreqs:
        for key in dictio:
            if streotype.lower() == key:
                value = dictio[key]
                book_name = wikinames[book_number]
                matches[book_name] = value
        book_number += 1
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_matches) > 10:
        sorted_matches = sorted_matches[0:10]
    return sorted_matches

# SEARCH FUNCTION WORKING AS THE 'MAIN' FUNCTION:

#Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():

    #Get query from URL variable (relevance: r_query, boolean: b_query)
    r_query = request.args.get('r_query')
    b_query = request.args.get('b_query')

    #Initialize list of matches
    matches = []
    
    #If relevance search query exists
    if r_query:
        # Search non-stemmed or stemmed articles based on the query's use of quotes
        # Receive the results as a dictionary
        if re.search(r'^".*"$', r_query):
            try:
                r_query = re.sub(r'^"(.*)"$', r'\1', r_query)
                results = search_file(r_query)
                articles_dict = wordfreq(wikidoc)
                frequencies = freqhits(articles_dict, r_query) ## tää sana korvataan inputilla 
                print(frequencies)  # tää on täällä koska ohjelma jää muuten looppaamaan ja html-sivu ei toimi
                                    # poistetaan sit joskus kun saadaan plotit toimimaan kunnolla
                seabornplot(frequencies) ##tää piirtää varsinaisen kuvan
                # Make the wikipedia article dictionary refer to the results dictionary
                # so that the html code can reference the results' entries
                wikidictionary = results
                # Make the matches list refer to the wikidictionary
                # so that the html code can reference the results' entries
                matches = wikidictionary
            except IndexError:
                matches.append("Your query \"{:s}\" didn't match any documents.".format(r_query))
        else:
            try:
                results = search_file_stem(r_query)
                articles_dict = wordfreq(wikidoc)
                frequencies = freqhits(articles_dict, r_query) ## tää sana korvataan inputilla 
                print(frequencies)  # tää on täällä koska ohjelma jää muuten looppaamaan ja html-sivu ei toimi
                                    # poistetaan sit joskus kun saadaan plotit toimimaan kunnolla
                seabornplot(frequencies) ##tää piirtää varsinaisen kuvan
                wikidictionary = results
                matches = wikidictionary
            except IndexError:
                matches.append("Your query '{:s}' didn't match any documents.".format(r_query))
                
    #If boolean search query exists
    if b_query:
        
        #Initialize list of match results
        results = []
        
        query = test_query(b_query)
        sparse_td_matrix = sparse_matrix.T.tocsr()
        
        try:
            hits_matrix = eval(query)
        
        # If there's a syntax error, append it into the matches list and
        # immediately return that list for the html page so its if statement works properly
        except SyntaxError:
            matches.append("SyntaxError: If your search consists of multiple words, remember to separate them with 'and' or 'or'. Also make sure you write 'and/or not' instead of just 'not'.")
            return render_template('index.html', matches=matches)
            
        hits_list = list(hits_matrix.nonzero()[1])
        
        # Make the boolean search results into dictionary entries for name, number of hits, and rank,
        # and append them into the results list
        i = 0   # using a counter here cause i'm a noob lel
        if len(hits_list) == 0:
            matches.append("Your query '{:s}' didn't match any documents.".format(b_query))     # no dictionary to make the html if statement work properly
            return render_template('index.html', matches=matches)
        elif len(hits_list) > 10:
            #results.append("Your query '{:s}' matches {:d} documents. The first ten matches are:".format(b_query, len(hits_list)))
            resultsitem = {'name': "Article Title", 'text': "Your query '{:s}' matched {:d} articles. Here are the first 25 words of the first ten articles:".format(b_query, len(hits_list)), 'rank': "#"}
            results.append(resultsitem)
            for doc_idx in hits_list[0:10]:
                resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'rank': i+1}
                results.append(resultsitem)
                i = i+1
        elif len(hits_list) == 1:
            #results.append("Your query '{:s}' matches the following document:".format(b_query))
            resultsitem = {'name': "Article Title", 'text': "Your query '{:s}' matched one article. Here are the first 25 words of it:".format(b_query), 'rank': "#"}
            results.append(resultsitem)
            for doc_idx in hits_list:
                resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'rank': i+1}
                results.append(resultsitem)
                i = i+1
        else:
            #results.append("Your query '{:s}' matches the following {:d} documents:".format(b_query, len(hits_list)))
            resultsitem = {'name': "Article Title", 'text': "Your query '{:s}' matched {:d} articles. Here are the first 25 words of them:".format(b_query, len(hits_list)), 'rank': "#"}
            results.append(resultsitem)
            for doc_idx in hits_list:
                resultsitem = {'name': wikinames[doc_idx], 'text': first_25_words(wikidoc[doc_idx]), 'rank': i+1}
                results.append(resultsitem)
                i = i+1
                
        articles_dict = wordfreq(wikidoc)
        frequencies = freqhits(articles_dict, b_query) ## tää sana korvataan inputilla 
        print(frequencies)  # tää on täällä koska ohjelma jää muuten looppaamaan ja html-sivu ei toimi
                            # poistetaan sit joskus kun saadaan plotit toimimaan kunnolla
        seabornplot(frequencies) ##tää piirtää varsinaisen kuvan
        
        # Make the wikipedia article dictionary refer to the results dictionary
        # so that the html code can reference the results' entries
        wikidictionary = results
        # Make the matches list refer to the wikidictionary
        # so that the html code can reference the results' entries
        matches = wikidictionary
        
    # Part of the original tutorial code is here as a point of reference:

    #If query exists (i.e. is not None)
    #if query:
        #Look at each entry in the example data
        #for entry in wikidoc:
            #If an entry name contains the query, add the entry to matches
            #if query.lower() in entry.lower():
                #matches.append(entry)

    #Render index.html with matches variable
    return render_template('index.html', matches=matches)
