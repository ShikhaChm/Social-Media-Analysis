#Sentiment Analysis
  #Dependencies
  import tweepy
  import json
  import numpy as np
  import pandas as pd
  from datetime import datetime
  import matplotlib.pyplot as plt
  from matplotlib import style

  from config import consumer_key, consumer_secret, access_token, access_token_secret
  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

  #Setup Tweepy API Authentication
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
  analyzer = SentimentIntensityAnalyzer()
  
  newsHouses  = ("@BBCNews", "@CBSNews", "@CNNnewsroom", "@FoxNews", "@nytimes")
  
  #array for sentiments 
  sentiments = []
  numTweets= 100
  
  for media in newsHouses:
      counter = 1
      #Get all tweets from home feed
      newsTweets = api.user_timeline(media, count= numTweets)

      #Loop through all tweets 
      for tweet in newsTweets:

          Print Tweets
          print("Tweet %s: %s" % (counter, tweet["text"]))

          #run Vader Analysis on each tweet
          results = analyzer.polarity_scores(tweet["text"])
          compound = results["compound"]
          pos = results["pos"]
          neu = results["neu"]
          neg = results["neg"]
          tweetsAgo = counter

          #add sentiments for each tweet into an array
          sentiments.append({"Date": tweet["created_at"], 
                             "Source": media,
                             "Compound": compound,
                             "Positive": pos,
                             "Negative": neg,
                             "Neutral": neu,
                             "Text": tweet['text'],
                             "Tweets Ago": counter})

          counter = counter + 1  
          
      #Put sentiments data into DataFrame 
      sentimentsDf = pd.DataFrame.from_dict(sentiments)
      sentimentsDf.to_csv("tweetsNews.csv", encoding= 'utf-8', index = False)
      sentimentsDf.head(10)
      
      colors = ['green','red','cyan','blue','yellow']
      looper = np.arange(0,len(newsHouses))  

## Scatter plot of the tweets sentiment analysis using dataframe sorted by time of tweet 
      for i in looper:
          newsChannel = sentimentsDf.loc[sentimentsDf["Source"] == newsHouses[i]]
          newsChannel = newsChannel.sort_values("Tweets Ago")
          plt.scatter(np.arange(len(newsChannel["Compound"])), newsChannel['Compound'],
                  color = colors[i], marker="o", alpha=0.8, linewidths=1, edgecolor="black", label= newsHouses[i]) 

      now = datetime.now().strftime("%m-%d-%Y  %H:%M")
      plt.title("Sentiment Analysis of Media Tweets " + now)
      plt.xlabel("Tweets Ago")
      plt.ylabel("Tweet Polarity")
      plt.grid(True)

      L = plt.legend(bbox_to_anchor=(1.05,1),title="Media Sources", loc= 2, borderaxespad = 0.)
      L.get_texts()[0].set_text('BBC')
      L.get_texts()[1].set_text('CBS')
      L.get_texts()[2].set_text('CNN')
      L.get_texts()[3].set_text('Fox')
      L.get_texts()[4].set_text('New York Times')

      plt.savefig("ScatterPlotNewsTweets.png")
      plt.show()
      
## Bar plot of the tweets sentiment analysis using dataframe sorted by time of tweet 
      newsMediaTweets = {}

      for i in looper:
          newsChannel = sentimentsDf.loc[sentimentsDf["Source"] == newsHouses[i]]
          newsMediaTweets[newsHouses[i]] = newsChannel["Compound"].mean()

      ind = np.arange(len(newsHouses))
      width = 0.35
      fig, ax = plt.subplots()    
      now = datetime.now().strftime("%m-%d-%Y  %H:%M")
      ticks = []

      plt.title("Overall Media Sentiment Based on Twitter " + now)
      plt.ylabel("Tweet Polarity")
      plt.xlim(-0.25, len(newsHouses))
      plt.ylim(min(newsMediaTweets.values()) - .005, max(newsMediaTweets.values()) + .05)

      for i in ind:
          ticks.append(i + 0.4)

      colors = ['green','red','cyan','skyblue','yellow']

      #add labels, title and axes ticks
      plt.xticks(ticks, newsMediaTweets, rotation= "vertical")
      barPlot = plt.bar(ind, newsMediaTweets.values(), color = colors, alpha=0.75, align="edge")

      def autolabel(rects, xpos='center'):
          """Attach a text label above each bar displaying its height"""
          for rect in rects:
              height = rect.get_height()
              ax.text(rect.get_x() + rect.get_width()/2., .08*height,
                       '-%.4f' % float(height),
                      ha='center', va='bottom', fontsize=8)

      autolabel(barPlot, "center")
      plt.show()
