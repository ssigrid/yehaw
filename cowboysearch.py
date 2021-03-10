from flask import Flask, render_template, request
import nltk
import re
import numpy as np
import spacy
import pke
import string
import math
import csv
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from sys import platform

#Initialize Flask instance
app = Flask(__name__)

# BOOK EDITING FUNCTIONS:

def list_newlines(file):

    """ Splits the source file with the songs into a list based on the "</song>" tag,
        deletes the <song name="[insert name]"> tags from the list,
        leaves the newlines as they are
    """

    openfile = open(file, "r", encoding='utf-8')
    readfile = openfile.read()
    songlist = readfile.split("</song>")
    # Delete an empty line from the end of the list:
    songlist = songlist[:-1]
    # Delete song name tags:
    i = 0
    for onesong in songlist:
        songlist[i] = re.sub(r'\n?<song name="[^"]+">', r'', onesong)
        # Delete the symbol at the beginning of the file
        songlist[i] = re.sub(r'\ufeff', r'', songlist[i])
        i = i + 1
    openfile.close()

    return songlist     # returns a list where every song is an item on said list
    
def list_titles(songfile):

    """ Splits the source file with the songs into a list based on the "</song>" tag,
        deletes the newline characters and
        substitutes the <song name="[insert name]"> tag with the name in it
    """
    
    openfile = open(songfile, "r", encoding='utf-8')
    readfile = openfile.read()
    songlist = readfile.split("</song>")
    # Delete an empty line from the end of the list:
    songlist = songlist[:-1]
    # Delete song name tags:
    i = 0
    for onesong in songlist:
        songlist[i] = re.sub(r'\n', r'', onesong)
        songlist[i] = re.sub(r'\n?<song name="([^"]+)">', r'\1', songlist[i])
        i = i + 1
    openfile.close()

    return songlist     # returns a list where every song is an item, name included but without newline characters
    
# STEMMING FUNCTIONS:

def stemmer(stringoftext):
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    porter = PorterStemmer()

    token_words = word_tokenize(stringoftext)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def stem_list(listofsongs):

    """ Runs every line on the list through a stemmer
        and writes the stemmed line in a new list
    """
    
    stemlist = []
    for song in listofsongs:
        stemlist.append(stemmer(song))

    return stemlist     # returns a list where every song is an item, name included but without newline characters, text has been stemmed

# GET THE SONG NAME FOR RESULTS:

def get_song_name(i):

    """ Splits the source file with the songs into a list based on the "</song>" tag,
        uses the given interger (i) to turn the list into an item
        that only contains the song whose name we want to get,
        searches said item using BeautifulSoup's html parser for a tag called 'song',
        then searches inside that song tag for a 'name' attribute,
        and returns the line of string that the name attribute is attached to
    """

    openfile = open(songs, "r", encoding='utf-8')
    readfile = openfile.read()
    songlist = readfile.split("</song>")
    # Delete an empty line from the end of the list:
    songlist = songlist[i]
    openfile.close()
    
    soupsong = BeautifulSoup(songlist, 'html.parser')
    for songtag in soupsong.find_all('song'):
        songname = songtag.get('name')
    
    return songname     # returns the name of the song as string

# TOPIC RANK FUNCTION:

def topic_rank(plotlines):
    
    """ Extracts words that are most important to the text
        based on the given part-of-speech variable and the index number of the song
    """
    
    # Create a TopicRank extractor
    extractor = pke.unsupervised.TopicRank()
    
    # Load the content of the song
    extractor.load_document(input=plotlines, language='en', normalization=None)
    
    # Create a list of stopwords and select candidates
    # based on what checkboxes the search page user selected (e.g. 'PROPN')
    # TopicRank selects the longest sequences of the given parts-of-speech as candidates
    # and ignores punctuation marks or stopwords as candidates
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=None, stoplist=stoplist)
    
    # Build topics by grouping candidates with HAC (average linkage, threshold of 1/4 of shared stems)
    # Weight the topics using random walk, and select the first occuring candidate from each topic
    extractor.candidate_weighting(threshold=0.74, method='average')
    
    # Get the 10-highest scored candidates as keyphrases
    # Keyphrases come in tuples (iirc) where the first one is the keyword/phrase and the second one is its score
    keyphrases = extractor.get_n_best(n=10)
    
    return keyphrases   # return the keyphrases and their scores as a list of tuples

# PLOTTING FUNCTION:

def plotting(query, plotlines, j):
    """ creates a cvs file from the ten keywords and keyword scores of the song j,
        returns a seaborn striplot """
    mlp.use("Agg")

    top10 = topic_rank(plotlines)
    row_list = [["score", "keyword"]]
    for t in top10:
        keyword = t[0]
        score = t[1]
        line = [score, keyword]
        row_list.append(line)
        
    with open("songs.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter= ',')
        writer.writerows(row_list)

    sns.set_theme(style="dark")
    
    dataset = pd.read_csv("songs.csv")

    # Make the PairGrid
    g = sns.PairGrid(dataset.sort_values("score", ascending=False),
                    x_vars=None, y_vars=["keyword"],
                    height=5, aspect=1)
    

    # Draw a dot plot using the stripplot function
    g.map(sns.stripplot, size=10, orient="h", jitter=False,
        palette="crest", linewidth=1, edgecolor="w")

    # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(0), xlabel="Relevance", ylabel="")

    for ax in g.axes.flat:

    # Set a different title for each axes
        #ax.set(title=title)

    # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    
    # Rewrite symbols that cannot be in image url
    if "*" in query:
        query = query.replace("*", "")
    if '"' in query:
        query = query.replace('"', '')
    if " " in query:                    # technically space is not an unacceotable symbol
        query = query.replace(" ", "_") # but spaces in file names can sometimes cause problems
    
    # Make j into string so it can be used in image url
    new_j = str(j)
    
    # Image url
    pltpath = f'{query}_{new_j}_plot.png'
    
    # Save the figure
    try:
        g.savefig(f'static/{pltpath}')
        
    # OSError happens if there is an unacceptable symbol in the url,
    # in which case the pltpath leads to a crying cowboy
    except OSError:
        pltpath = "cowboy_crying.png"
    
    return pltpath

# RELEVANCE SEARCH FUNCTION:

def relevance_songs(query, t_rank):

    """ Relevance search (non-stem, non-n-gram)
    """
    
    # Initialize list of results
    results = []

    if "*" in query:
        # Vectorize query string
        wildcard_vec = wildcard_songs(query)
        if wildcard_vec == None:
            results.append("Your query \"{:s}\" didn't match any documents.".format(query))
            return results
        else:
            # Cosine similarity
            hits = np.dot(wildcard_vec, g3_matrix)
            # Rank hits
            ranked_scores_and_doc_ids = \
            sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    elif re.match(r'^"(.*)"$', query):
        query_noquotes = re.sub(r'^"(.*)"$', r'\1', query)
        gv1 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
        g_matrix = gv1.fit_transform(songs_nonewlines).T.tocsr()
        query_vec = gv1.transform([ query_noquotes ]).tocsc()
        hits = np.dot(query_vec, g_matrix)
        ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    else:
        query_stem = stemmer(query)
        gv2 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
        g_matrix_stem = gv2.fit_transform(songs_stemmed).T.tocsr()
        query_vec = gv2.transform([ query_stem ]).tocsc()
        hits = np.dot(query_vec, g_matrix_stem)
        ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]), reverse=True)
    
    # Output result
    # Make the relevance search results into dictionary entries
    # where name=song name, text=lyrics, score=relevance search score, rank=number of song in the results,
    # and append them into the results list:
    
    # The first entry is different and has 'title' as a key instead of 'text' so it gets printed properly on the html page,
    # what exactly the title key says depends on the length of the ranked hit (i.e. how many hits the query has)
    if len(ranked_scores_and_doc_ids) == 1:
        resultsitem = {'name': "Song Title", 'title': "Your query {:s} matched one song. Here are its lyrics:".format(query), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        j = 1
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)   # we're only printing four decimals now
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)   # getting rid of some extra newlines
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            if t_rank == True:
                plotlines = songs_nonewlines[doc_idx]
                pltpath = plotting(query, plotlines, j)
                j = j + 1
            else:
                pltpath = ""
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'score': roundedscore, 'rank': i+1, 'plotimg': pltpath}
            results.append(resultsitem)
    elif len(ranked_scores_and_doc_ids) <= 10:
        resultsitem = {'name': "Song Title", 'title': "Your query {:s} matched {:d} songs. Here are their lyrics in order of relevance:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        j = 1
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            if t_rank == True:
                plotlines = songs_nonewlines[doc_idx]
                pltpath = plotting(query, plotlines, j)
                j = j + 1
            else:
                pltpath = ""
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'score': roundedscore, 'rank': i+1, 'plotimg': pltpath}
            results.append(resultsitem)
    else:
        resultsitem = {'name': "Song Title", 'title': "Your query {:s} matched {:d} songs. Here are the lyrics of the 10 songs your query matched the best:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        j = 1
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 9:
                roundedscore = "{:.4f}".format(score)
                newlines = songs_newlines[doc_idx]
                newlines = re.sub(r'\n\n\n', r'\n', newlines)
                newlines = re.sub(r'\n\n\n', r'\n', newlines)
                newlines = newlines.split("\n")
                if len(newlines[0]) == 0:
                    newlines = newlines[1:] # no empty line at the beginning
                if t_rank == True:
                    plotlines = songs_nonewlines[doc_idx]
                    pltpath = plotting(query, plotlines, j)
                    j = j + 1
                else:
                    pltpath = ""
                resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'score': roundedscore, 'rank': i+1, 'plotimg': pltpath}
                results.append(resultsitem)
    return results      # return dictionary of results

def wildcard_songs(query):
    global g3_matrix
    
    query_len = len(query.split())
    gv3 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", ngram_range=(query_len, query_len))
    g3_matrix = gv3.fit_transform(songs_nonewlines).T.tocsr()
    terms3 = gv3.get_feature_names()

    # Replace the wildcard with a regex pattern matching 0 or more characters:
    #query_no_wc = query.replace("*", ".*")
    query_replace_wc = re.sub(r'\*','.*', query)

    # Compile a regex pattern based on the query word:
    wildcard_pattern = re.compile(query_replace_wc)

    # Find all matching words in the vocabulary and form a new query word list:
    matching = [w for w in terms3 if re.fullmatch(wildcard_pattern, w)]
    if matching:
        new_query_string = " ".join(matching)
        wildcard_vec = gv3.transform([new_query_string]).tocsc()
    else:
        return None

    return wildcard_vec

# BOOLEAN SEARCH FUNCTIONS:

def rewrite_token(t):
    global td_matrix
    global t2i
    
    cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
    sparse_matrix = cv.fit_transform(songs_nonewlines)
    dense_matrix = sparse_matrix.todense()
    td_matrix = dense_matrix.T
    terms = cv.get_feature_names()
    t2i = cv.vocabulary_
    
    d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements
    
    # Testing if the query is in terms (which contains all words in the searched material):
    if t not in terms:
        # Operators are left in place but other queries without hits get replaced with zeroes:
        if t not in d:
            return "np.zeros((1, len(songs_nonewlines)), dtype=int)"   # returns an array filled with zeroes that has the length of the number of songs
    
    # Queries that do have hits get the matrix treatement:
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split()) # splits the query into words and returns the rewritten, rejoined query

def boolean_songs(b_query, hits_list, t_rank):
    
    """ Boolean search (exact match, non-n-gram)
    """
    
    # Initialize list of results
    results = []
    
    # Make the boolean search results into dictionary entries
    # where name=song name, text=lyrics, rank=number of song in the results,
    # and append them into the results list:
    
    # The first entry is different and has 'title' as a key instead of 'text' so it gets printed properly on the html page,
    # what exactly the title key says depends on the length of the ranked hit (i.e. how many hits the query has)
    i = 0
    if len(hits_list) == 1:
        resultsitem = {'name': "Song Title", 'title': "Your query {:s} matched one song. Here are its lyrics:".format(b_query), 'rank': "#"}
        results.append(resultsitem)
        j = 1
        for doc_idx in hits_list:
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)   # getting rid of some extra newlines
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            if t_rank == True:
                plotlines = songs_nonewlines[doc_idx]
                pltpath = plotting(b_query, plotlines, j)
                j = j + 1
            else:
                pltpath = ""
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'rank': i+1, 'plotimg': pltpath}
            results.append(resultsitem)
            i = i+1
    elif len(hits_list) <= 10:
        resultsitem = {'name': "Song Title", 'title': "Your query {:s} matched {:d} songs. Here are their lyrics:".format(b_query, len(hits_list)), 'rank': "#"}
        results.append(resultsitem)
        j = 1
        for doc_idx in hits_list:
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            if t_rank == True:
                plotlines = songs_nonewlines[doc_idx]
                pltpath = plotting(b_query, plotlines, j)
                j = j + 1
            else:
                pltpath = ""
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'rank': i+1, 'plotimg': pltpath}
            results.append(resultsitem)
            i = i+1
    else:
        resultsitem = {'name': "Song Title", 'title': "Your query {:s} matched {:d} songs. Here are the lyrics of the first ten songs:".format(b_query, len(hits_list)), 'rank': "#"}
        results.append(resultsitem)
        j = 1
        for doc_idx in hits_list[0:10]:
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            if t_rank == True:
                plotlines = songs_nonewlines[doc_idx]
                pltpath = plotting(b_query, plotlines, j)
                j = j + 1
            else:
                pltpath = ""
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'rank': i+1, 'plotimg': pltpath}
            results.append(resultsitem)
            i = i+1

    return results      # return dictionary of results

# SOURCE FILE GETTING HANDLED:

songs = "cowboy_songs.txt"
songs_newlines = list_newlines(songs)           # this is needed for printing the songs on the html page, it has newlines
songs_nonewlines = list_titles(songs)           # this is where the query search happens, it has no newlines but it does have song titles
songs_stemmed = stem_list(songs_nonewlines)     # this is where the query stem search happens, it has no newlines but it does have song titles
cowboydictionary = []                           # it's a surprise tool that will help us later :)
                                                # (i.e. there needs to exist a dictionary even before searches for the html to work for some reason)

# SEARCH FUNCTION WORKING AS THE 'MAIN' FUNCTION:

#Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():
	
    # Delete previous plots and .csv files
    if platform == "win32":
        try:
            imglist = glob.glob('static/*_plot.png')
            for imgpath in imglist:
                os.remove(imgpath)
            os.remove('songs.csv')
        except:
            pass
    else:
        try:
            os.system('rm -f static/*_plot.png')
            os.system('rm -f songs.csv')
        except:
            pass
    
    # Get query from URL variable
    # Actual query: query, relevance button: r_query, boolean button: b_query
    query = request.args.get('query')
    r_query = request.args.get('r_query')
    b_query = request.args.get('b_query')
    
    #Get TopicRank from URL variable
    t_rank = request.args.get('topicrank')
    
    # If TopicRank checkbox has been checked, t_rank is True
    if t_rank:
        t_rank = True
    else:
        t_rank = False

    #Initialize list of matches
    matches = []
    
    # If query exists
    if query:
        # If both or neither the relevance or boolean searches has been selected,
        # append error message to the matches list
        if not r_query and not b_query or r_query and b_query:
            matches.append("Please check either boolean or relevance search.")
        # If relevance search query exists
        elif r_query:
            # Try relevance search, append an error message if it doesn't work
            # Receive the results as a dictionary
            try:
                results = relevance_songs(query, t_rank)
                # Make the cowboydictionary refer to the results dictionary
                # so that the html code can reference the results' entries:
                cowboydictionary = results
                # Make the matches list refer to the cowboydictionary
                # so that the html code can reference the results' entries:
                matches = cowboydictionary
            except IndexError:
                    matches.append("Your query {:s} didn't match any documents.".format(query))
                
        # If boolean search query exists
        elif b_query:
        
            #Initialize list of match results
            results = []
        
            rewritten_query = rewrite_query(query)
        
            try:
                hits_matrix = eval(rewritten_query)
            # If there's a syntax error, append it into the matches list and
            # immediately return that list for the html page so its if statement works properly
            except SyntaxError:
                matches.append("SyntaxError: If your search consists of multiple words, remember to separate them with 'and' or 'or'. Also make sure you write 'and/or not' instead of just 'not'.")
                return render_template('index.html', matches=matches)
            
            hits_list = list(hits_matrix.nonzero()[1])
        
            if len(hits_list) == 0:
                matches.append("Your query {:s} didn't match any songs.".format(query))     # no dictionary to make the html if statement work properly
                return render_template('index.html', matches=matches)
            else:
                results = boolean_songs(query, hits_list, t_rank)
                cowboydictionary = results
                matches = cowboydictionary

    #Render index.html with matches variable
    return render_template('index.html', matches=matches)