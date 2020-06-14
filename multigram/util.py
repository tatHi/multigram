from datetime import datetime
import random
import pickle

def getTimeStamp():
    # timestamp
    ts = datetime.today()
    timeStamp = ts.strftime('%Y%m%d%H%M%S')

    # randomID
    source_str = 'abcdefghijklmnopqrstuvwxyz'
    randomID = ''.join([random.choice(source_str) for x in range(10)])
    timeStamp += '-'+randomID
    return timeStamp

