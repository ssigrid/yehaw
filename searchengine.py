#näitä tehtävänjakoja voi poistaa kun on tehty

# miten avata tiedosto linkistä eli ohjeiden kohta 5
    # tiedoston lukeminen
    # tiedoston jako artikkeleihin
    # artikkelien jako stringeiksi
    # tiedoston sulkeminen


# kohta 2 ohjeista
# kohta 3 ohejista
# kohta 4 ohjeista


# mainin toiminta? jos on aikaa



# Pistin tän lukeen tekstin suoraan tiedostosta koska Moodlen suojaukset
# (ainakin luulisin että niissä syy?) estää tekstin lukemisen urlista.
# Deletöin myös erillisen close file -funktion koska se hoituu yhdellä komennolla.
def read_file(file):

    """ Function that opens the file, reads it, and writes the data in it
    into a list where every wiki article is its own item in the list
    """

    openfile = open(file, "r")
    readfile = openfile.read()
    filelist = readfile.split("</article>")
    openfile.close()
    
    return filelist     # returns the data that's been split into a list
    
wiki = "wiki100.txt"    # This needs to be added into the main function if/when we have one
#test = read_file(wiki) # Delete hashtagged commands later, this was just
#print(test[0])         # to test if the function works as it's supposed to


def rewrite_token(t):
    d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t))

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()

from sklearn.feature_extraction.text import CountVectorizer


documents = ["This is a silly example",
             "A better example",
             "Nothing to see here",
             "This is a great and long example"]

wikidoc = read_file(wiki)
cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(wikidoc)

dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_


loop = True
while loop == True:
    print("Input query or 0 to quit: ")
    query = input()
    if query == "0":
        print("Goodbye!")
        loop = False
    else:
        test_query(query)
        ## siirsin nää alemmat tänne koska muuten printtasivat myös sulkiessa
        ## varmain main ohjelmaan tai omaansa sit lopussa
        sparse_td_matrix = sparse_matrix.T.tocsr()
        hits_matrix = eval(rewrite_query("NOT example OR great")) 
        hits_list = list(hits_matrix.nonzero()[1])
        for doc_idx in hits_list:
            print("Matching doc:", wikidoc[doc_idx])
        for i, doc_idx in enumerate(hits_list):
            print("Matching doc #{:d}: {:s}".format(i, wikidoc[doc_idx]))

