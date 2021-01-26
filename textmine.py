def main():

    """ Main program that takes the text from the Yle news page dedicated to Coronavirus,
        searches the text belonging to the CSS class meant for Coronavirus news,
        then prints all text in that class that has the HTML tag h3 or h6.
    """

    import nltk, re, pprint             # I don't know if these are all necessary to import
    from nltk import word_tokenize      # but I put them here just in case
    from urllib import request
    from bs4 import BeautifulSoup
    
    url = "https://yle.fi/uutiset/18-89760"
    url_decoded = request.urlopen(url).read().decode('utf8')
    souptext = BeautifulSoup(url_decoded, 'html.parser')
    ronanews = souptext.find("div", class_="GridSystem__GridCell-sc-15162af-0 fUiKQy")
    for headline in ronanews.find_all(['h3', 'h6']):
        print(headline.string)
        print()
    
main()