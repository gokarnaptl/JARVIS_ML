#Function
import datetime
from Speak import Say
#2 Types

#1 - Non Input
#eg: Time ,Data , Speedtest

def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    Say(time)

def Date():
    date = datetime.data.today()
    Say(date)

def Day():
    day = datetime.datetime().now().strftime("%A")
    Say(day)

def NonInputExecution(query):

    query = str(query)

    if"time" in query:
        Time()

    elif "date" in query:
        Data()

    elif"day" in query:
        Day()

#2 - Input
#eg - google search , wikipedia

def InputExecution(tag,query):

    if "wikipedia" in tag:
        name = str(query).replace("who is","").replace("about","")
        import wikipedia
        result = wikipedia.summary(name)
        Say(result)

    elif "google" in tag:
        query = str(query).replace("google","")
        query = query.replace("search","")
        import pywhatkit
        pywhatkit.search(query)

