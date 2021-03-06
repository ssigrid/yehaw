# yehaw
This is the NLP project for the course KIK-LG211 by Sanna, Viima, and Sigrid. 


#### Using flask
##### Install Flask in a virtual environment to contain it.


Mac - Create a virtual environment hawyeenv:
> python3 -m venv hawyeenv


Windows:
>py -3 -m venv hawyeenv


Mac - Activate the environment:
>. hawyeenv/bin/activate


Windows:
>hawyeenv/Scripts/activate


NOTE: If Windows command line can't run the command, move into the Scripts directory and run:
>activate


You need to have python3 in order for the code to work. Install also Flask, nltk, sklearn, seaborn, spacy, bs4, pandas and numpy using pip3, in this way:
>pip3 install Flask

>pip3 install nltk

>pip3 install sklearn

>pip3 install seaborn

>pip3 install spacy

>pip3 install bs4

>pip3 install pandas

>pip3 install numpy


Then install pke, and somet nltk tags and stuff:

>pip3 install git+https://github.com/boudinfl/pke.git

>python3 -m nltk.downloader stopwords

>python3 -m nltk.downloader universal_tagset

>python3 -m spacy download en


##### Run this Flask application


Make sure you are in yehaw directory.


Every time you open the environment, set the following environment variables:


Mac - Show flask which file to run, enable development environment to activate interactive debugger and reloader, and set the port in which to run the application:
>export FLASK_APP=cowboysearch.py

>export FLASK_ENV=development

>export FLASK_RUN_PORT=8000


On Windows command line, you can the environment variables with:
>set FLASK_APP=cowboysearch.py

>set FLASK_ENV=development

>set FLASK_RUN_PORT=8000


And on Windows PowerShell:
>$env:FLASK_APP = "cowboysearch.py"

>$env:FLASK_ENV = "development"

>$env:FLASK_RUN_PORT = "8000"


Run the app:
>flask run


Go to localhost:8000/search in your browser to see the website.
