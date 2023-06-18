import streamlit as st
import json
from pickle import load
import json
import mysql.connector

st.title('Diagnosing Disease')
st.header('You can diagnose your disease here and we can give you the result!')

st.header('Type your symptoms')
labell = 'e.g I have a headache'
symptom = st.text_input(labell)
prohibit = [labell, ""]

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="snatia"
)

def preprocess_sentences(string):

    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    from re import findall

    string = string.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    string = tokenizer.tokenize(string)

    stemer = SnowballStemmer(language='english')
    string = [stemer.stem(i) for i in string]

    stopword    = set(stopwords.words('english'))
    string = [i for i in string if i not in stopword]

    string = " ".join(string)

    string = findall('[a-z]+', string)

    return string



keluhan = ['I am so anxious and i often panic',
           'I am so stressed because of the problem which hit me',
           'dad has been coughing for 4 hour',
           'my gf cant sleep at night and she has been sleepless for 2 days',
           'my sister needs to loss some weight']

def run():

    c = load(open('count_vec.pkl', 'rb'))
    t = load(open('tfid.pkl', 'rb'))
    l = load(open('label.pkl', 'rb'))
    m = load(open('model.pkl', 'rb'))

    data = json.load(open('data.json', 'r'))
    lenn = json.load(open('len.json', 'r'))

    sent = data['keluhan']
    sent = preprocess_sentences(sent)
    sent = " ".join(sent)
    prediksi = l.inverse_transform(m.predict(t.transform(c.transform([sent]))))[0]

    predict = {"prediksi":prediksi}

    id = lenn['panjang'] + 1

    mycursor = mydb.cursor()
    sql = """INSERT INTO prediksi(`id_keluhan`, `keluhan`, `prediksi`) VALUES (%s, %s, %s)"""
    val = (id, data['keluhan'], prediksi)

    json.dump({"panjang":id}, open('len.json', 'w'), indent=2)

    mycursor.execute(sql, val)
    mydb.commit()

    json.dump(predict, open('prediksi.json', 'w'), indent=2)

    result = json.load(open('prediksi.json', 'r'))

    st.header(f'Prediction result : {result["prediksi"]}')

if st.button('Klik'):

    if symptom not in prohibit:
        js = {'keluhan':symptom}
        json.dump(js, open('data.json', 'w'), indent=2)
        run()