import sys
import csv
import tweepy
import re 
import nltk
import string
from nltk.classify import *
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from nltk.corpus import stopwords
import nltk.classify.util

#initialize stopWords
stopWords = []
 
#starting the function 
def replaceTwoOrMore(s):
   
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end
 
#starting the function 
def getStopWordList(stopWordListFileName):
#read the stopwords file and build a list
    stopWords = []
    #stopWords.append('TWITTER_USER')
    stopWords.append('URL')
 
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

st = open('StopWords.txt', 'r')
stopWords = getStopWordList('StopWords.txt')
 
#starting the function 
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end
 
#starting the function 
def featureExtraction():
    
    filepath="MlKhattarsent.csv"
    with open(filepath, "r",encoding='utf-8', errors='ignore') as csvfile:
        inpTweets = csv.reader(csvfile)
        tweets = []
        for row in inpTweets:
            sentiment = row[7]
            tweet = row[1]
            featureVector = getFeatureVector(tweet)
            tweets.append((featureVector, sentiment))
    
    return tweets 
#end

tweets = featureExtraction()


#Classifier 
def get_words_in_tweets(tweets):
    all_words = []
    for (text, sentiment) in tweets:
        all_words.extend(text)
    return all_words

def get_word_features(wordlist):
    
    # This line calculates the frequency distrubtion of all words in tweets
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    
   
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets)) #my list of many words 

def extract_features(tweet):
    settweet = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in settweet)
    return features




training_set = nltk.classify.apply_features(extract_features, tweets)
test_set = nltk.classify.apply_features(extract_features, tweets[:250])



classifier = nltk.NaiveBayesClassifier.train(training_set)


accuracy = nltk.classify.accuracy(classifier, training_set) 


print (accuracy) 

total = accuracy * 100 
print ('Naive Bayes Accuracy: %4.2f' % total) 


accuracyTestSet = nltk.classify.accuracy(classifier, test_set) 

 
print (accuracyTestSet) 

totalTest = accuracyTestSet * 100 
print ('\nNaive Bayes Accuracy with the Test Set: %4.2f' % totalTest) 
