# 1. Import Dependencies
import re
import csv
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

# 2. Setup Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# 3. Setup Pipeline
topics = ['TATA']

# 4.1. Search for Stock News using Google and Yahoo Finance
print('Searching stock news for', topics)
def search_urls(topic):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(topic)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs

raw_urls = {topic:search_urls(topic) for topic in topics}

# 4.2. Strip out unwanted URLs
print('Removing unwanted urls...')
excludewords = ['maps', 'policies', 'preferences', 'accounts', 'support']
def remove_urls(urls, excludewords):
    valid = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in excludewords):
            result = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            valid.append(result)
    return list(set(valid))

cleaned_urls = {topic:remove_urls(raw_urls[topic], excludewords) for topic in topics} 

# 4.3. Search and Scrape Cleaned URLs
print('Scraping through cleaned urls...')
def search_scrape(links):
    contents = []
    for link in links: 
        r = requests.get(link)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [parah.text for parah in paragraphs]
        words = ' '.join(text).split(' ')[:300]
        content = ' '.join(words)
        contents.append(content)
    return contents

article = {topic:search_scrape(cleaned_urls[topic]) for topic in topics} 

# 4.4. Summarise all Articles
print('Summarizing articles...')
def summarize(article):
    summaries = []
    for item in article:
        input_item = tokenizer.encode(item, return_tensors='pt')
        output = model.generate(input_item, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {topic:summarize(article[topic]) for topic in topics}

# 5. Adding Sentiment Analysis
print('Calculating sentiment...')
sentiment = pipeline("sentiment-analysis")
scores = {topic:sentiment(summaries[topic]) for topic in topics}

# # 6. Exporting Results
print('Exporting results to csv file...')
def create_array(summaries, scores, urls):
    output = []
    for topic in topics:
        for counter in range(len(summaries[topic])):
            output_items = [
                topic,
                summaries[topic][counter],
                scores[topic][counter]['label'],
                scores[topic][counter]['score'],
                urls[topic][counter]
            ]
            output.append(output_items)
    return output
final = create_array(summaries, scores, cleaned_urls)
final.insert(0, ['Topic', 'Summary', 'Sentiment', 'Confidence', 'URL'])

with open('sentiments.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final)