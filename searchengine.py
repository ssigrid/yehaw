# kohta 4 ohjeista


# mainin toiminta? jos on aikaa

def read_file(file):

    """ Function that opens the file, reads it, and writes the data in it
    into a list where every wiki article is its own item in the list
    """

    openfile = open(file, "r", encoding='utf-8')
    print("Writing the file into a list...")
    readfile = openfile.read()
    filelist = readfile.split("</article>")
    filelist = filelist[:-1]                        # Deletes an empty line from the end of the list
    openfile.close()
    print("Success!")
    print()
    
    return filelist     # returns the data that's been split into a list

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
    
    #print("Query: '" + query + "'")
    #print("Rewritten:", rewrite_query(query))
    
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
    # print("Matching:", eval(r_join))
    
    return r_join   # returns the rewritten list
    print()

def first_20_words(article): ## this now prints the first twenty fords of the article, including the name
    import re
    first_20 = re.sub(r'<article name="(([A-Za-z \(\)0-9])+)">', r'\1 –', article)
    word_list = first_20.split()
    for i in range(21):
        print(word_list[i], end=" ")
    print("...")

from sklearn.feature_extraction.text import CountVectorizer

wiki = "wiki100.txt"
wikidoc = read_file(wiki)

cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(wikidoc) ## tää on oikee mut testailen
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T
terms = cv.get_feature_names()
t2i = cv.vocabulary_


def main():
    
    """ Main program that asks the user to input a query, then searches that query
    from the articles, and prints the first 20 words from the first ten matching articles
    """
    
    loop = True
    while loop == True:
        print("Input query or an empty string to quit: ")
        query = input()
        if query == "":
            print("Goodbye!")
            loop = False
        else:
            query = test_query(query)
            sparse_td_matrix = sparse_matrix.T.tocsr()
            
            try:
                hits_matrix = eval(query)
            except SyntaxError:
                print("SyntaxError: If your search consists of multiple words, remember to separate them with 'and' or 'or'.")
                print("Also make sure you write 'and/or not' instead of just 'not'.")
                print()
                continue
            
            hits_list = list(hits_matrix.nonzero()[1])
            print("Matches:", len(hits_list))
            print()

            if len(hits_list) == 0:
                print("No matches for query.")
                print()
            elif len(hits_list) > 10:
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

main()