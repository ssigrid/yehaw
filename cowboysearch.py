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

    # List where the rewritten list items go (any other way of doing this messed other functions up for some reason)
    rewrittensongs = []
    
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

def relevance_songs(query):

    """ Relevance search (non-stem, non-n-gram)
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
    # Make the relevance search results into dictionary entries
    # where name=song name, text=lyrics, score=relevance search score, rank=number of song in the results,
    # and append them into the results list:
    
    # The first entry is different and has 'title' as a key instead of 'text' so it gets printed properly on the html page,
    # what exactly the title key says depends on the length of the ranked hit (i.e. how many hits the query has)
    if len(ranked_scores_and_doc_ids) == 1:
        resultsitem = {'name': "Song Title", 'title': "Your query \"{:s}\" matched one song. Here are its lyrics:".format(query), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)   # we're only printing four decimals now
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)   # getting rid of some extra newlines
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    elif len(ranked_scores_and_doc_ids) <= 10:
        resultsitem = {'name': "Song Title", 'title': "Your query \"{:s}\" matched {:d} songs. Here are their lyrics in order of relevance:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            roundedscore = "{:.4f}".format(score)
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
    else:
        resultsitem = {'name': "Song Title", 'title': "Your query \"{:s}\" matched {:d} songs. Here are the lyrics of the 10 songs your query matched the best:".format(query, len(ranked_scores_and_doc_ids)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            if 0 <= i <= 9:
                roundedscore = "{:.4f}".format(score)
                newlines = songs_newlines[doc_idx]
                newlines = re.sub(r'\n\n\n', r'\n', newlines)
                newlines = re.sub(r'\n\n\n', r'\n', newlines)
                newlines = newlines.split("\n")
                if len(newlines[0]) == 0:
                    newlines = newlines[1:] # no empty line at the beginning
                resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'score': roundedscore, 'rank': i+1}
                results.append(resultsitem)
          
    return results      # return dictionary of results

# BOOLEAN SEARCH FUNCTIONS:

def rewrite_token(t):
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

def boolean_songs(b_query, hits_list):
    
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
        resultsitem = {'name': "Song Title", 'title': "Your query '{:s}' matched one song. Here are its lyrics:".format(b_query), 'rank': "#"}
        results.append(resultsitem)
        for doc_idx in hits_list:
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)   # getting rid of some extra newlines
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'rank': i+1}
            results.append(resultsitem)
            i = i+1
    elif len(hits_list) <= 10:
        resultsitem = {'name': "Song Title", 'title': "Your query '{:s}' matched {:d} songs. Here are their lyrics:".format(b_query, len(hits_list)), 'rank': "#"}
        results.append(resultsitem)
        for doc_idx in hits_list:
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'rank': i+1}
            results.append(resultsitem)
            i = i+1
    else:
        resultsitem = {'name': "Song Title", 'title': "Your query '{:s}' matched {:d} songs. Here are the lyrics of the first ten songs:".format(b_query, len(hits_list)), 'rank': "#"}
        results.append(resultsitem)
        for doc_idx in hits_list[0:10]:
            newlines = songs_newlines[doc_idx]
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = re.sub(r'\n\n\n', r'\n', newlines)
            newlines = newlines.split("\n")
            if len(newlines[0]) == 0:
                newlines = newlines[1:] # no empty line at the beginning
            resultsitem = {'name': get_song_name(doc_idx), 'text': newlines, 'rank': i+1}
            results.append(resultsitem)
            i = i+1

    return results      # return dictionary of results

def single_rank(pos, songnumber):
    
    """ Extracts words that are most important to the text
        based on the given part-of-speech variable and the index number of the song
    """
    
    # Initialize list of results
    results = []

    # Find the song to analyze from the list where every song is one item without newlines
    singlesong = songs_nonewlines[songnumber]
    
    # Create SingleRank extractor
    extractor = pke.unsupervised.SingleRank()
    
    # Load the content of the song
    extractor.load_document(input=singlesong, language='en', normalization=None)
    
    # Select candidates based on what checkboxes the search page user selected (e.g. 'PROPN')
    # SingleRank selects the longest sequences of the given pos as candidates
    extractor.candidate_selection(pos=pos)
    
    # Weight the candidates using the sum of their word's scores that are computed using random walk
    # In the graph (in case we draw it), nodes are words of certain part-of-speech
    #that are connected if they occur in a window of 10 words
    extractor.candidate_weighting(window=10, pos=pos)
    
    # Get the 10-highest scored candidates as keyphrases
    # Keyphrases come in tuples (iirc) where the first one is the keyword/phrase and the second one is its score
    keyphrases = extractor.get_n_best(n=10)
    
    # Make the keyphrase search results into dictionary entries
    # where name=keyword/keyphrase, title=TBA, score=score, rank=number of song in the results,
    # and append them into the results list:
    if len(keyphrases) == 0:
        # In the rare case where there are no results, no dictionary entry is created:
        results.append("No hits.")
    else:
        i = 0
        resultsitem = {'name': "Keyword(s)", 'title': "Song number {:d} is called {:s}.".format(songnumber+1, get_song_name(songnumber)), 'score': "Score", 'rank': "#"}
        results.append(resultsitem)
        for hit in keyphrases:
            roundedscore = "{:.4f}".format(hit[1])
            resultsitem = {'name': hit[0], 'title': "This is where we would put an example of the keyword in the song.", 'score': roundedscore, 'rank': i+1}
            results.append(resultsitem)
            i = i+1
    
    return results

# SOURCE FILE GETTING HANDLED:

songs = "cowboy_songs.txt"
songs_newlines = list_newlines(songs)   # this is needed for printing the songs on the html page, it has newlines
songs_nonewlines = list_titles(songs)   # this is where the query search happens, it has no newlines but it does have song titles
cowboydictionary = []                   # it's a surprise tool that will help us later :)
                                        # (i.e. there needs to exist a dictionary even before searches for the html to work for some reason)

# VECTORS AND MATRICES:

cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(songs_nonewlines)
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_

gv1 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix = gv1.fit_transform(songs_nonewlines).T.tocsr()
gv2 = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2", token_pattern=r"(?u)\b\w+\b")
g_matrix_stem = gv2.fit_transform(songs_nonewlines).T.tocsr()


def plotting_data(song):
    """ creates data for the seaborn so that y = (theme)words and x = their tf-idf score
    """
    tf = {}
    tfidf = {}
    fdist = nltk.FreqDist(w.lower() for w in nltk.corpus.gutenberg.words(song))
    for w, f in fdist.most_common():
        tf[w] = 1 + math.log10(f)
        tfidf[w] = tf[w] * idf[w]
    tfidf_sorted_words = soted(tfidf.keys(), key=lambda w: tfidf[w], reverse=True)
    top10 = tfidf_sorted_words[0:10]
    return top10

#data = plotting_data(song)         ## this would mean the song
# def plotting(data):
    
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
    
    #Get index number of the song for SingleRank analysis from URL variable
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
    
    # Finding out what the final pos variable will be based on which x_pos variables exist
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
    
    # If the user inputed a number and checked one of the pos checkboxes, run single_rank to get results
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