For the local execution of the demo, it is necessary to have flask and Python 3 installed. The operating system for which this execution is intended is Linux (it has been tested on Manjaro).

Once Flask is installed, it is required to create a virtual environment and activate it as follows:
```
python3 -m venv venv
. venv/bin/activate
```

The next step is to install the necessary dependencies to be able to run the python Backend.
```
pip install -r requirements.txt
```

Finally, run the program in flask:
```
flask run app.py
```