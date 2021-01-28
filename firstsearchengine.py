# selvittää looppiin tarvitaan
        # pyytää antamaan test_query tyylisen sisällön? ja nollalla lopettaa
# miten avata tiedosto linkistä
# tiedoston lukeminen
# tiedoston jako artikkeleihin
# artikkelien jako stringeiksi
# tiedoston sulkeminen
# kohta 2 ohjeista
# kohta 3 ohejista
# kohta 4 ohjeista
# kohta 5 ohjeista
## tarviiko main ohjelmaa ollenkaan vaikka se tuolla nyt on
# sijottaa sijotettavat minne kuuluu




## tehään tää kans
# def read_file(file):
# def close_file(file):


def rewrite_token(t):
    d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements
    return d.get(t, 'td_matrix[t2i["{:s}"]]'.format(t)) # Can you figure out what happens here?

def rewrite_query(query): # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())

def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query))) # Eval runs the string as a Python command
    print()

from sklearn.feature_extraction.text import CountVectorizer



    ## oletan et read_file pitää määrätä täällä, mut voi olla et pitää eka kysyy kumpi tiedosto
documents = ["This is a silly example",
             "A better example",
             "Nothing to see here",
             "This is a great and long example"]

cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)

dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_


    # looppi toimii,
    # mutta mitä queryä tähän piti laittaa
loop = True
while loop == True:
    print("Input query or 0 to quit: ")
    query = input()
    if query == "0":
        loop = False
    else:
        test_query(query)
print("Quitting...")        
print("Goodbye!")

sparse_td_matrix = sparse_matrix.T.tocsr()
hits_matrix = eval(rewrite_query("NOT example OR great")) 
hits_list = list(hits_matrix.nonzero()[1])
for doc_idx in hits_list:
    print("Matching doc:", documents[doc_idx])
for i, doc_idx in enumerate(hits_list):
    print("Matching doc #{:d}: {:s}".format(i, documents[doc_idx]))
