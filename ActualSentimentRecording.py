import csv
from textblob import TextBlob
import codecs
import re
with open('MlKhattarsent.csv','w') as cvfile:
    writer = csv.writer(cvfile)
    with open ('mlshift.csv', 'r') as csvfile:
        reader =csv.reader(csvfile)
        for row in reader:
            tweet=row[1].strip()
            cleanTweet=" ".join(re.findall("[a-zA-Z]+",tweet))
            analysis= TextBlob(cleanTweet)
            if analysis.sentiment.polarity > 0:
                writer.writerow([row[0], row[1], row[2],row[3],row[4],row[5], analysis.sentiment.polarity, 4])
            elif analysis.sentiment.polarity <0:
                writer.writerow([row[0], row[1], row[2],row[3],row[4],row[5], analysis.sentiment.polarity, 0])
            #else:
                #writer.writerow([row[0], row[1], row[2],row[3],row[4],row[5], analysis.sentiment.polarity, "neutral"])
                