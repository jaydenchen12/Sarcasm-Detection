import re
from twython import Twython
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
def filter(text):
    if ('\n' in text):
        return False
    else:
        if ('!' in text):
            return True
        elif ('...' in text):
            return True
        elif ('?' in text):
            return True
        else:
            return False
#Variables that contains the user credentials to access Twitter API
access_token = "932358974036430848-paIDtBwyeJrVJYbHVemqvgEPogSZjMN"
access_token_secret = "2LbAsPSPcQFnYfwzwudWd6Hdzh9B8oRvwrSIH7hwLzMo4"
consumer_key = "HKr8Al8oWC3boyPqJ36VXIZ5B"
consumer_secret = "2RZUsBn87Fhm7IQzlwohsV3gMybor7vkkMo7AVg8vgHoCq0T4U"

t = Twython(app_key=consumer_key,
            app_secret=consumer_secret,
            oauth_token=access_token,
            oauth_token_secret=access_token_secret)
query = "#fact  -filter:retweets AND -filter:replies"
search = t.search(q=query,   #**supply whatever query you want here**
                  count=4000)

tweets = search['statuses']

for tweet in tweets:
    if (len(tweet['text']) > 30 and filter(tweet['text'])):
        text = re.sub('#fact', ' ', tweet['text'])
        text = re.sub('#','', text)
        text = re.sub('http(\S+)\s?', '', text)
        print (tweet['id_str'], '\t', text, '\t\t\t')


#######################################################################
query = "#sarcastic  -filter:retweets AND -filter:replies"
search = t.search(q=query,   #**supply whatever query you want here**
                  count=4000)

tweets = search['statuses']

for tweet in tweets:
    if (len(tweet['text']) > 30 and filter(tweet['text'])):
        text = re.sub('#sarcastic', '', tweet['text'])
        text = re.sub('http(\S+)\s?', '', text)
        text = re.sub('#','', text)
        print (tweet['id_str'], '\t', text, '\t\t\t')


#########################################################################
query = "#eyeroll  -filter:retweets AND -filter:replies"
search = t.search(q=query,   #**supply whatever query you want here**
                  count=4000)

tweets = search['statuses']

for tweet in tweets:
    if (len(tweet['text']) > 30 and filter(tweet['text'])):
        text = re.sub('#eyeroll', '', tweet['text'])
        text = re.sub('#','', text)
        text = re.sub('http(\S+)\s?', '', text)
        print (tweet['id_str'], '\t', text, '\t\t\t')



