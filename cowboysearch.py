from flask import Flask, render_template, request
import nltk
import re
import numpy as np
import spacy
import pke
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

#Initialize Flask instance
app = Flask(__name__)

# this splits the songs into a list:

def one_item_per_song(file):

    """ Explain this
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
        i = i + 1
    openfile.close()

    return songlist     # returns the file split into a list where every song is an item
    
def one_line_per_song(file):

    """ Explain this
    """

    openfile = open(file, "r", encoding='utf-8')
    readfile = openfile.read()
    songlist = readfile.split("</song>")
    # Delete an empty line from the end of the list:
    songlist = songlist[:-1]
    i = 0
    for onesong in songlist:
        songlist[i] = re.sub(r'\n', r'', onesong)
        i = i + 1
    openfile.close()

    return songlist     # returns the file split into a list where every song is an item on one line

def get_song_name(line):

    """ Explain this
    """

    soupsong = BeautifulSoup(line, 'html.parser')
    for songtag in soupsong.find_all('song'):
        songname = songtag.get('name')
    
    return songname

def relevance_songs(query):

    """ Explain this
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
    # Make the relevance search results into dictionary entries for song title, number of hits, and rank,
    # and append them into the results list
    if len(ranked_scores_and_doc_ids) == 1:
        resultsitem = {'name': "Song Title", 'title': "Your query \"{:s}\" matched one song. Here are its lyrics:".format(query), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)   # we're only printing four decimals now
            newlines = song_item[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(song_line[doc_idx]), 'text': newlines, 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    elif len(ranked_scores_and_doc_ids) <= 10:
        resultsitem = {'name': "Song Title", 'title': "Your query \"{:s}\" matched {:d} songs. Here are their lyrics in order of relevance:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)
            newlines = song_item[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(song_line[doc_idx]), 'text': newlines, 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    else:
        resultsitem = {'name': "Song Title", 'title': "Your query \"{:s}\" matched {:d} songs. Here are the lyrics of the 10 songs your query matched the best:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 9:
                roundedscore = "{:.4f}".format(score)
                newlines = song_item[doc_idx]
                newlines = re.sub(r'\n\n\n', r'\n', newlines)
                newlines = re.sub(r'\n\n\n', r'\n', newlines)
                newlines = newlines.split("\n")
                if len(newlines[0]) == 0:
                    newlines = newlines[1:] # no empty line at the beginning
                resultsitem = {'name': get_song_name(song_line[doc_idx]), 'text': newlines, 'score': roundedscore, 'rank': i+1}
                results.append(resultsitem)
          
    return results      # return dictionary of results

# BOOLEAN SEARCH FUNCTIONS:

def rewrite_token(t):
    d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements
    
    # Testing if the query is in the Wikipedia articles:
    if t not in terms:
        # Operators are left in place but other queries without hits get replaced with zeroes:
        if t not in d:
            return "np.zeros((1, len(song_line)), dtype=int)"   # returns an array filled with zeroes that has the length of the number of wikipedia articles
    
    # Queries that do have hits get the matrix treatement:
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split()) # splits the query into words and returns the rewritten, rejoined query

def boolean_songs(b_query, hits_list):
    
    """ Explain this
    """
    
    # Initialize list of results
    results = []
    
    # Make the boolean search results into dictionary entries for name, number of hits, and rank,
    # and append them into the results list
    i = 0
    if len(hits_list) == 1:
        resultsitem = {'name': "Song Title", 'title': "Your query '{:s}' matched one song. Here are its lyrics:".format(b_query), 'rank': "#"}
        results.append(resultsitem)
        for doc_idx in hits_list:
            newlines = song_item[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(song_line[doc_idx]), 'text': newlines, 'rank': i+1}
            results.append(resultsitem)
            i = i+1
    elif len(hits_list) <= 10:
        resultsitem = {'name': "Song Title", 'title': "Your query '{:s}' matched {:d} songs. Here are their lyrics:".format(b_query, len(hits_list)), 'rank': "#"}
        results.append(resultsitem)
        for doc_idx in hits_list:
            newlines = song_item[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(song_line[doc_idx]), 'text': newlines, 'rank': i+1}
            results.append(resultsitem)
            i = i+1
    else:
        resultsitem = {'name': "Song Title", 'title': "Your query '{:s}' matched {:d} songs. Here are the lyrics of the first ten songs:".format(b_query, len(hits_list)), 'rank': "#"}
        results.append(resultsitem)
        for doc_idx in hits_list[0:10]:
            newlines = song_item[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(song_line[doc_idx]), 'text': newlines, 'rank': i+1}
            results.append(resultsitem)
            i = i+1

    return results      # return dictionary of results

def single_rank(pos, songnumber):    
    
    """ Explain this
    """
    
    # Initialize list of results
    results = []

    singlesong = song_item[songnumber]
    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=singlesong, language='en', normalization=None)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = extractor.get_n_best(n=10)
    
    if len(keyphrases) == 0:
        results.append("No hits.")
    else:
        i = 0
        resultsitem = {'name': "Keyword(s)", 'title': "Song number {:d} is called {:s}.".format(songnumber+1, get_song_name(song_line[songnumber])), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for hit in keyphrases:
            roundedscore = "{:.4f}".format(hit[1])
            newlines = song_item[songnumber]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': hit[0], 'title': "This is where we would put an example of the keyword in the song.", 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
            i = i+1
    
    return results

# SOURCE FILE GETTING HANDLED:

songs = "cowboy_songs.txt"
song_item = one_item_per_song(songs)
song_line = one_line_per_song(songs)
cowboydictionary = []

# VECTORS AND MATRICES:

cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(song_line)
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_

gv1 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix = gv1.fit_transform(song_line).T.tocsr()
gv2 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix_stem = gv2.fit_transform(song_line).T.tocsr()

# SEARCH FUNCTION WORKING AS THE 'MAIN' FUNCTION:

#Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():
	
    #Get query from URL variable (relevance: r_query, boolean: b_query)
    r_query = request.args.get('r_query')
    b_query = request.args.get('b_query')
    
    #Get part-of-speech from URL variable
    n_pos = request.args.get('noun')
    propn_pos = request.args.get('propn')
    adj_pos = request.args.get('adj')
    posnum = request.args.get('posnum')

    #Initialize list of matches
    matches = []
    
    #If relevance search query exists
    if r_query:
        # Search non-stemmed or stemmed articles based on the query's use of quotes
        # Receive the results as a dictionary
        if re.search(r'^".*"$', r_query):
            try:
                r_query = re.sub(r'^"(.*)"$', r'\1', r_query)
                results = relevance_songs(r_query)
                # Make the cowboydictionary refer to the results dictionary
                # so that the html code can reference the results' entries:
                cowboydictionary = results
                # Make the matches list refer to the cowboydictionary
                # so that the html code can reference the results' entries:
                matches = cowboydictionary
            except IndexError:
                matches.append("Your query \"{:s}\" didn't match any documents.".format(r_query))
        else:
            try:
                results = relevance_songs(r_query) # stemmed search comes here later
                cowboydictionary = results
                matches = cowboydictionary
            except IndexError:
                matches.append("Your query '{:s}' didn't match any documents.".format(r_query))
                
    #If boolean search query exists
    if b_query:
        
        #Initialize list of match results
        results = []
        
        query = rewrite_query(b_query)
        sparse_td_matrix = sparse_matrix.T.tocsr()
        
        try:
            hits_matrix = eval(query)
        # If there's a syntax error, append it into the matches list and
        # immediately return that list for the html page so its if statement works properly
        except SyntaxError:
            matches.append("SyntaxError: If your search consists of multiple words, remember to separate them with 'and' or 'or'. Also make sure you write 'and/or not' instead of just 'not'.")
            return render_template('index.html', matches=matches)
            
        hits_list = list(hits_matrix.nonzero()[1])
        
        if len(hits_list) == 0:
            matches.append("Your query '{:s}' didn't match any songs.".format(b_query))     # no dictionary to make the html if statement work properly
            return render_template('index.html', matches=matches)
        else:
            results = boolean_songs(b_query, hits_list)
            cowboydictionary = results
            matches = cowboydictionary
    
    if n_pos:
        if propn_pos:
            if adj_pos:
                pos = {'NOUN', 'PROPN', 'ADJ'}
            else:
                pos = {'NOUN', 'PROPN'}
        elif adj_pos:
            pos = {'NOUN', 'ADJ'}
        else:
            pos = {'NOUN'}
        if not posnum:
            matches.append("Please input the number of the song you want to analyze.")
    elif propn_pos:
        if adj_pos:
            pos = {'PROPN', 'ADJ'}
        else:
            pos = {'PROPN'}
        if not posnum:
            matches.append("Please input the number of the song you want to analyze.")
    elif adj_pos:
        pos = {'ADJ'}
        if not posnum:
            matches.append("Please input the number of the song you want to analyze.")
    
    if posnum:
        songnumber = int(posnum)
        songnumber = songnumber - 1
        if n_pos or propn_pos or adj_pos:
            results = single_rank(pos, songnumber)
            cowboydictionary = results
            matches = cowboydictionary
        else:
            matches.append("Please check one of the boxes.")

    #Render index.html with matches variable
    return render_template('index.html', matches=matches)