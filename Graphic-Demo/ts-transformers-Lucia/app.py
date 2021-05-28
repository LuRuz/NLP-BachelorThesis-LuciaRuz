# import my_summary as ms                  # Specify the path within Env. variables

# 
# python3 -m venv venv 
# . venv/bin/activate
# pip install -r requirements.txt 
# flask run


import re
import requests                                    
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

import lucia_sum as ls

app = Flask("__name__")
paragraphs = []

@app.route('/')
def index():
    return render_template('index.html', setText="Here your summary...")


@app.route('/', methods=['POST'])
def getvalue():
    text_input = request.form['name']
    selection = request.form['selection']

    if text_input=="":
        rem = "Error, empty form"
        return render_template('index.html', rem=rem)
    else:
        summary = ls.text_summarization(selection,text_input)
        title = "Summary generated with "+selection+" model."
        return render_template('index.html', setText=summary, setTitle=title)



if __name__ == '__main__':
    app.run(debug=True)
