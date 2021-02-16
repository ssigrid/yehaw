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
    
def first_25_words(article):
    first_25 = re.sub(r'<article name="(([A-Za-z \(\)0-9])+)">', r'', article)
    word_list = first_25.split()
    twentyfivewords = []
    for i in range(26):
        twentyfivewords.append(word_list[i])
    twentyfivewords = ' '.join(twentyfivewords)
    twentyfivewords = twentyfivewords + "..."
    
    return twentyfivewords      # return the first 25 words of the article as string

wiki = "wiki100.txt"
wikidoc = read_file(wiki)
wikinames = get_name(wikidoc)
wikistem = stem_file(wikidoc)
wikidictionary = []

for i in range(100):    # initializing the dictionary entries might be redundant, check later
    dictionarynew = {'name': wikinames[i], 'text': first_25_words(wikidoc[i])}
    wikidictionary.append(dictionarynew)

#Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():

    #Get query from URL variable
    query = request.args.get('query')

    #Initialize list of matches
    matches = []

    #If query exists (i.e. is not None)
    if query:
        #Look at each entry in the example data
        for entry in wikidoc:
            #If an entry name contains the query, add the entry to matches
            if query.lower() in entry.lower():
                matches.append(entry)

    #Render index.html with matches variable
    return render_template('index.html', matches=matches)
