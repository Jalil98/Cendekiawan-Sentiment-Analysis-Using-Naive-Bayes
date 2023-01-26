# Importing essential libraries
from flask import Flask, render_template, request, session, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle
import os
import csv
import base64
from io import BytesIO
import string 
import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
plt.rcParams['figure.figsize'] = 12, 8
import uuid
secret_key = uuid.uuid4().hex

app = Flask(__name__)

# Define secret key to enable session
app.secret_key = secret_key

# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load the model NB and vectorizer
filename = 'Tuning-Model-NB.pkl'
classifier = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))


@app.route('/', methods=['GET'])
def home():
	return render_template('home.html')


# Function for preprocessing text
def text_preprocessing(text): 
    text = remove_text_special(text)
    text = remove_emoji(text)
    # Tokenizing
    tokens = word_tokenize(text)
    # Removing stop words
    stopwords = stopword_removal(tokens)
    # Create Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in stopwords]
    return " ".join(stemmed_tokens).strip().lower()


# Remove special text
def remove_text_special(text):
    # Remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '',str(text))
    # Replace kata yang berulang-ulang ('oooooo' menjadi '00')
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Replace 2+ dots with space
    text = re.sub(r'\.{2,}', ' ', text)
    # Remove @username
    text = re.sub('@[^\s]+','',text)
    # Remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # Remove angka
    text = re.sub('[0-9]+', '', text)
    # Remove url
    text = re.sub(r"http\S+", "", text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Strip space, " and ' from tweet
    text = text.strip(' "\'')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans("","",string.punctuation))
    # Remove character 
    text = text.replace("\n",' ').replace("Diterjemahkan oleh Google",' ').replace("Asli",' ')
    # Remove url uncomplete
    return text.replace("http://", " ").replace("https://", " ")


# Remove emoticon
def remove_emoji(text):
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    encoded_string = text.encode("ascii", "ignore")
    text = encoded_string.decode()

    return(text)

# Stopword Function
def stopword_removal(text):
    filtering = stopwords.words('indonesian')
    x = []
    data = []
    def myFunc(x):
        if x in filtering:
            return False
        else:
            return True
    fit = filter(myFunc, text)
    for x in fit:
        data.append(x)
    return data

# Function to make a wordcloud
def render_word_cloud(corpus):
    '''Generates a word cloud using all the words in the corpus.
    '''
    fig_file = BytesIO()
    wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='black', colormap = 'Dark2', stopwords=STOPWORDS).generate(corpus)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(fig_file, format='png')
    fig_file.seek(0)
    fig_data_png = fig_file.getvalue()
    result = base64.b64encode(fig_data_png)
    return result.decode('utf-8')


# Function to make a bar graph
def bar_graph(dataframe=None):
    '''Generates bar graph using comments dataframe.
    '''
    fig_file = BytesIO()
    x = dataframe.prediction
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_palette('Set2')
    class_names = ['negatif', 'netral', 'positif']
    bar = sns.countplot(x=x)
    bar.set(xticklabels=[])  
    bar.set_xticklabels(class_names)
    bar.set(xlabel=None)
    plt.savefig(fig_file, format='png')
    fig_file.seek(0)
    fig_data_png = fig_file.getvalue()
    result = base64.b64encode(fig_data_png)
    return result.decode('utf-8')


# Function to make a donut chart that will use in pie graph
def donut(sizes, ax, angle=90, labels=None,colors=None, explode=None, shadow=None):
    ax.pie(sizes, colors = colors, labels=labels, autopct='%.1f%%', 
           startangle = angle, pctdistance=0.8, explode = explode, 
           wedgeprops=dict(width=0.4), shadow=shadow)
    plt.axis('equal')  
    plt.tight_layout()


# Function to make a pie graph
def pie_graph(dataframe=None):
    '''Generates pie graph using comments dataframe.
    '''
    fig_file = BytesIO()
    df = dataframe
    sizes = dataframe.prediction.value_counts()
    labels = ['negatif', 'netral', 'positif']
    colors = ['#FF0000', '#50C878', '#3521cf']
    explode = (0,0,0)

    fig, ax = plt.subplots(figsize=(6,4))
    donut(sizes, ax, 90, labels, colors=colors, explode=explode, shadow=True)
    plt.savefig(fig_file, format='png')
    fig_file.seek(0)
    fig_data_png = fig_file.getvalue()
    result = base64.b64encode(fig_data_png)
    return result.decode('utf-8')

@app.route('/predict-text', methods=['POST'])
def predict_text():
    if request.method == 'POST':
        input_text = request.form['input_text']
        text_prep = text_preprocessing(input_text)
        text_matrix = vectorizer.transform([text_prep])
        pred = classifier.predict(text_matrix.toarray())
        proba = classifier.predict_proba(text_matrix.toarray())

        if pred.argmax() == 0 :
            text = 'Negatif'
            class_proba = int(proba[0][0].round(2)*100)
            
        elif pred.argmax() == 1 :
            text = 'Netral'
            class_proba = int(proba[0][1].round(2)*100) 
          
        elif pred.argmax() == 2 :
            text = 'Positif'
            class_proba = int(proba[0][2].round(2)*100) 
        
        return render_template('home.html', result_pred=text, result_proba=class_proba)
        
    

@app.route('/predict-file', methods=['POST'])
def predict_file():
    if request.method == 'POST':
        uploaded_df = request.files['file']
        data_filename = secure_filename(uploaded_df.filename)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        filepath = session['uploaded_data_file_path']

        dict_data = []
        with open(filepath, encoding='utf-8') as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                dict_data.append(row)

        review = pd.DataFrame()
        for row in dict_data:
            row['preprocessing'] = text_preprocessing(row['reply'])
            v_data = vectorizer.transform([row['reply']]).toarray()
            row['prediction'] = str(classifier.predict(v_data)).strip("[]").strip("'")
            review = review.append(row, ignore_index=True)
        
        review = review[['reply', 'preprocessing', 'prediction']]

        df_wc_pos = review[review['prediction'] == 'positif']
        df_wc_neg = review[review['prediction'] == 'negatif']
        df_wc_neu = review[review['prediction'] == 'netral']
        
        pos_string = []
        for t in df_wc_pos.preprocessing:
            pos_string.append(t)
        pos_string = pd.Series(pos_string).str.cat(sep=' ')

        neg_string = []
        for t in df_wc_neg.preprocessing:
            neg_string.append(t)
        neg_string = pd.Series(neg_string).str.cat(sep=' ')

        neu_string = []
        for t in df_wc_neu.preprocessing:
            neu_string.append(t)
        neu_string = pd.Series(neu_string).str.cat(sep=' ')

        return render_template('prediction.html', 
                                column_names=review.columns.values, 
                                row_data=list(review.values.tolist()), 
                                zip=zip,
                                res_1=render_word_cloud(pos_string),
                                res_2=render_word_cloud(neg_string),
                                res_3=render_word_cloud(neu_string),
                                bar_graph=bar_graph(review),
                                pie_graph=pie_graph(review)
                              )

if __name__ == '__main__':
	app.run(debug=True)