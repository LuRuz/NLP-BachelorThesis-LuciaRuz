# import my_summary as ms                  # Specify the path within Env. variables

# 
# python3 -m venv venv 
# . venv/bin/activate
# pip install -r requirements.txt 
# flask run


import re
import requests                                    # https://www.youtube.com/c/PythonProgramming4u
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
        # response = requests.get(str(url_str))
        # cap = re.findall('[0-9]+\.ece', url_str)
        # cap1 = re.findall('[0-9]+', cap[0])
        # divid = "content-body-14269002-" + cap1[0]
        # soup = BeautifulSoup(response.content, "html.parser")
        # layout = soup.find("div", attrs={"id": divid})
        # paragraphs = layout.find_all('p')
        # SentRank = ms.summa(paragraphs)
        # # def getvalues1(paragraphs):
        # sentGen = ms.gensima(paragraphs)
        # return render_template('index.html', sentGen=sentGen, SentRank=SentRank)

        summary = ls.text_summarization(selection,text_input)
        title = "Summarry generated with "+selection+" model."
        return render_template('index.html', setText=summary, setTitle=title)



if __name__ == '__main__':
    app.run(debug=True)
