import os
import io
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    '''
    This function iterates through each email and 
    returns a complete file path with email message
    '''
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            fileName = io.open(path, 'r', encoding='latin1')
            for line in fileName:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            fileName.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    '''
    This function loads the email dataset and creates the dataframe for training
    the model
    '''
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index = index)
    
def multinomialNBClassifier(mailDf):
    '''
    This function is the training module, where the multinomial naiive bayes model
    is trained with training email dataset
    '''
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(mailDf['message'].values)

    classifier = MultinomialNB()
    targets = mailDf['class'].values
    classifier.fit(counts, targets)
    return classifier

def loadEmailMessages():
    '''
    This function loads and create the dataframe for the email messages to be classified
    '''
    emailsDf = pd.read_csv('Z:/ML/DataScience/DataScience/emails/emailsToClassify')
        
    return emailsDf

def classifyMails(emailsDf,classifier):
    '''
    This function predicts whether a given mail is a spam or ham
    '''
    vectorizer = CountVectorizer()
    emailCounts = vectorizer.transform(emailsDf)
    predictions = classifier.predict(emailCounts)
    return predictions

def main():
    '''
    This is a mail spam-ham classifier that classifies a given email into spam
    or ham(not spam)
    '''
    #creates a dataframe with mail body and label
    mailDf = DataFrame({'message': [], 'class': []})

    mailDf = mailDf.append(dataFrameFromDirectory('Z:/ML/DataScience/DataScience/emails/spam', 'spam'))
    mailDf = mailDf.append(dataFrameFromDirectory('Z:/ML/DataScience/DataScience/emails/ham', 'ham'))
    classifier = multinomialNBClassifier()
    emailsDf = loadEmailMessages()

    predictions = classifyMails(emailsDf,classifier)
    predictions.to_csv('Z:/ML/DataScience/DataScience/emails/predictor.csv', sep = ',')
    

if __name__ == 'main':
    main()
