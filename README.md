# 💸 Financial Sentiments Scraper

## 🗒️ Summary

This project focuses on scraping sentiments from the web (Yahoo! Finance in particular) based on finance related topics such as Stocks/Tickers. The scraping is done using `requests` and `BeautifulSoup` library, a simplified and orthodox method. The scraped/parsed texts are in the form of long paragraphs which are summarized using a transformer from `huggingface` library trained specifically for financial texts, `pegasus` ([know more](https://huggingface.co/docs/transformers/model_doc/pegasus)). Then, sentiment analysis is carried out with the help of pipeline API. Eventually, the decoded output summary & sentiment scores are compiled and converted into an array and the results are exported to a CSV file.

- Access jupyter notebook: [notebook](content/notebook.ipynb)
- Access python script: [script](content/main.py)

## 📱 Example Output

| Topic  | Summary | Sentiment | Confidence |
| ------ | ------- | --------- | ---------- |
| TATA  | Tata Communications to gain industry-proven platform with strong capabilities and scale. | Positive | 0.99976 |
| HDFC  | Shareholders in HDFC will receive 42 shares of the bank for every 25 shares. | Positive | 0.55101 |
| HDFC  | HDFC to stop issuing debt, rely on bank deposits. Yields could fall about 10 basis points. | Negative | 0.97710 |

## 📑 Code Snippets

1. Search for urls:
```python
def search_urls(topic):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(topic)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs

raw_urls = {topic:search_urls(topic) for topic in topics}
```

2. Remove unwanted urls:
```python
excludewords = ['maps', 'policies', 'preferences', 'accounts', 'support']

def remove_urls(urls, excludewords):
    valid = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in excludewords):
            result = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            valid.append(result)
    return list(set(valid))

cleaned_urls = {topic:remove_urls(raw_urls[topic], excludewords) for topic in topics}
```

3. Search and scrape through cleaned urls:
```python
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
```

4. Summarize articles:
```python
def summarize(article):
    summaries = []
    for item in article:
        input_item = tokenizer.encode(item, return_tensors='pt')
        output = model.generate(input_item, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {topic:summarize(article[topic]) for topic in topics}
```

5. Sentiment analysis:
```python
from transformers import pipeline
sentiment = pipeline('sentiment-analysis')

scores = {topic:sentiment(summaries[topic]) for topic in topics}
```
