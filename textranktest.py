import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pke

def read_file(file):

    openfile = open(file, "r", encoding='utf-8')
    readfile = openfile.read()
    #filelist = readfile.split("</article>")
    #filelist = filelist[99]                 
    filelist = ''.join(readfile)
    openfile.close()
    

    return filelist
	
cowboy = "textrankdoc.txt"
cowboydoc = read_file(cowboy)
	
pos = {'PROPN'}
extractor = pke.unsupervised.SingleRank()
extractor.load_document(input=cowboydoc, language='en', normalization=None)
extractor.candidate_selection(pos=pos)
extractor.candidate_weighting(window=10, pos=pos)
keyphrases = extractor.get_n_best(n=10)

print(keyphrases)