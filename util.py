import json
import nltk
from nltk.corpus import stopwords
import re

stopwords=stopwords.words('english')
punctuations=set([',','.','?','!','-', '_'])
with_num=False
with_punc=False


def get_plain(content):
    content_plain = re.sub(r'[^\x00-\x7f]',r'',content.strip())
    if with_num==False:
        content_plain = re.sub("[0-9]", " ", content_plain)  # strip numbers
    if with_punc==False:
        content_plain = re.sub(r'[^\w\s]',' ',content_plain) # strip puntuaction
    content_plain = re.sub("\s\s+" , " ", content_plain) # shrink multiple whitespace into single whitespace
    return content_plain

# timestep 1: 2012-2013
# timestep 2: 2014-2015
# timestep 3: 2016-2018
def split_timesteps():
    t1=set(["2012", "2013"])
    t2=set(["2014", "2015"])
    t3=set(["2016", "2017", "2018"])

    filename ='News_Category_Dataset_v2.json'
    cnt=0
    fp =open(filename, 'r')
    fout1 =open("sample1.txt",'w')
    fout2 =open("sample2.txt",'w')
    fout3 =open("sample3.txt",'w')
    for line in fp:    
        data = json.loads(line.strip())
        headline = get_plain(data["headline"].lower())
        tokens=headline.strip().split()
        cleaned_headline=[]
        for w in tokens:
            if (w not in stopwords) and (w not in punctuations):
                cleaned_headline.append(w)
        if len(cleaned_headline) > 0:                
            text=' '.join(cleaned_headline)
            dtg = data["date"]
            year = dtg.split('-')[0]
            if year in t1:
                fout1.write(text+'\n')
            if year in t2:
                fout2.write(text+'\n')
            if year in t3:
                fout3.write(text+'\n')
            #print(cleaned_headline, text)
        cnt+=1

    fp.close()
    fout1.close()
    fout2.close()
    fout3.close()
    print(cnt)

split_timesteps()
