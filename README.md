# yehaw
This is the NLP project for the course KIK-LG211 by Sanna, Viima, and Sigrid. 


#### Using flask
##### Install Flask in a virtual environment to contain it.

Create a virtual environment hawyeenv:
> python3 -m venv hawyeenv

On Windows:
>py -3 -m venv hawyeenv

Activate the environment:
>. hawyeenv/bin/activate

On Windows:
>hawyeenv/Scripts/activate

Install Flask, nltk, sklearn, and numpy:
>pip install Flask
>pip install nltk
>pip install  sklearn
>pip install numpy


##### Run this Flask application

Make sure you are in yehaw directory.

Set the following environment variables:

Show flask which file to run:
>export FLASK_APP=flaskdemo.py

Enable development environment to activate interactive debugger and reloader:
>export FLASK_ENV=development

Set the port in which to run the application, e.g.:
>export FLASK_RUN_PORT=8000

On Windows command line, you can the environment variables with:
>set FLASK_APP=flaskdemo.py
>set FLASK_ENV=development
>set FLASK_RUN_PORT=8000

And on Windows PowerShell:
>$env:FLASK_APP = "flaskdemo.py"
>$env:FLASK_ENV = "development"
>$env:FLASK_RUN_PORT = "8000"

Run the app:
>flask run

Go to localhost:8000/search in your browser to see the website.
