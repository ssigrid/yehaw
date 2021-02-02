#näitä tehtävänjakoja voi poistaa kun on tehty

# miten avata tiedosto linkistä eli ohjeiden kohta 5
    # tiedoston lukeminen
    # tiedoston jako artikkeleihin
    # artikkelien jako stringeiksi
    # tiedoston sulkeminen

# kohta 3 ohejista
# kohta 4 ohjeista


# mainin toiminta? jos on aikaa


def read_file(file):

    """ Function that opens the file, reads it, and writes the data in it
    into a list where every wiki article is its own item in the list
    """

    openfile = open(file, "r")
    readfile = openfile.read()
    filelist = readfile.split("</article>")
    openfile.close()
    
    return filelist     # returns the data that's been split into a list
       # This needs to be added into the main function if/when we have one
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

def first_20_words(article): ## this now prints the first twenty fords of the article, including the name
    import re
    first_20 = re.sub(r'<article name="(([A-Za-z \(\)0-9])+)">', r'\1 –', article)
    word_list = first_20.split()
    for i in range(21):
        print(word_list[i], end=" ")
    print("...")

wiki = "wiki100.txt"
from sklearn.feature_extraction.text import CountVectorizer


documents = ["This is a silly example",
            "A better example",
            "Nothing to see here",
            "This is a great and long example"]

wikidoc = read_file(wiki)
cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(wikidoc) ## tää on oikee mut testailen
#sparse_matrix = cv.fit_transform(documents) ## TESTI: laita tää risuaidal piiloo ja ota ylemmästä pois
                                                ## alla on viel kaks lisää näitä

dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_


def main():
    
    loop = True
    while loop == True:
        print("Input query or an empty string to quit: ")
        query = input()
        if query == "":
            print("Goodbye!")
            loop = False
        else:
            test_query(query)
            sparse_td_matrix = sparse_matrix.T.tocsr()
            hits_matrix = eval(rewrite_query(query))
            # mallissa tässä oli jotai printtejä mitä en kopioinu   
            hits_list = list(hits_matrix.nonzero()[1])
            print("Matches:", len(hits_list))
            print()

            if len(hits_list) > 10:
                print("Here are the first ten articles:")
                print()
                for doc_idx in hits_list[0:10]:
                    print("Matching article:", end=" ")
                    article_name = first_20_words(wikidoc[doc_idx])
                    print()
            else:
                for doc_idx in hits_list:
                    print("Matching article:", end=" ")
                    article_name = first_20_words(wikidoc[doc_idx])
                    print()
            print()

main()
