### charts the words in files, and shows have many times which of given words has been used
### in which files/articles/books


## create planning which shows how many times which word is in the books that have the most of them
    ## should the lenght's of the books be considered?
    ## ready selected words or play as you wish

## words to look at in the final project: cowboy, horse, brave, pistol, lasso, cow, dog, milk, 
## whiskey/beer/scoths/alcohol, god, bible
test_set = [["list", "in", "a", "list"], ["hope", "this", "this", "it", "it", "does", "it"], ["should", "i", "take", "a", "break"], ["i", "am", "clad", "this", "is", "this", "not", "usually", "manual"], ["can", "this", "i", "get", "a", "yeehaw"]]
def seaborn(matches):
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
        for word in article: 
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1
        dict_articles.append(dictionary)
    return dict_articles


def freqhits(wordfreqs, streotype):
    matches = {}
    book_name = 0 ## merkkaa tällä hetkellä sitä mikä sit ois kirjan nimi, nyt vaan numero
    for dictio in wordfreqs:
        for key in dictio:
            if streotype == key:
                value = dictio[key]
                matches[book_name] = value
        book_name += 1
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_matches) > 10:
        sorted_matches = sorted_matches[0:9]
    return sorted_matches

dictionary = wordfreq(test_set)
freggies = freqhits(dictionary, "this")
#print(dictionary)
seaborn(freggies)
