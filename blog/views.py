from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import Post
from .forms import PostForm
from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
from collections import Counter, defaultdict
import pandas as pd
import concurrent.futures
import time
import random
import unicodedata
import spacy
import os
import zipfile
from django.conf import settings
import tempfile
import shutil
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def check_columns(request):
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(blog_post);")
        columns = [row[1] for row in cursor.fetchall()]
    return HttpResponse(", ".join(columns))


def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {'posts': posts})

def post_detail(request, slug):
    post = get_object_or_404(Post, slug=slug)
    return render(request, 'blog/post_detail.html', {'post': post})

def error_404_view(request, exception):
    data = {"name": 'Blog dla programistów'}
    return render(request, 'blog/404.html', data)

def post_new(request):
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})

def post_edit(request, slug):
    post = get_object_or_404(Post, slug=slug)
    # logika edycji posta
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm(instance=post)
    return render(request, 'blog/post_edit.html', {'form': form})


def post_delete(request, pk):
    post = get_object_or_404(Post, pk=pk)
    post.delete()
    return redirect('post_list')  # Przekierowuje użytkownika na listę postów po usunięciu


@csrf_exempt
def submit_landing_page(request):
    if request.method == 'POST':
        content = request.POST.get('content')
        # Przetwórz dane z formularza tutaj
        # Możesz dodać zapis do bazy danych, wysyłanie e-maila itp.
        return redirect('landing_page')
    return render(request, 'blog/landing_page.html')



# Załaduj model SpaCy do analizy języka polskiego
nlp = spacy.load("pl_core_news_sm")
nlp.max_length = 10000000  # Maksymalna długość tekstu przetwarzanego przez SpaCy

# Wczytaj listę polskich stopwordów
# Wczytaj listę polskich stopwordów z pliku
stopwords_file_path = os.path.join(os.path.dirname(__file__), 'polish.stopwords.txt')
with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    polish_stopwords = set(file.read().split())

def is_polish_page(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang') == 'pl':
            return True
        text = soup.get_text().lower()
        polish_words = ['jest', 'się', 'i', 'w', 'z', 'nie', 'na', 'że', 'to', 'co', 'jak']
        if any(word in text for word in polish_words):
            return True
    except Exception as e:
        print(f"Error checking URL {url}: {e}")
    return False

def search_google(query, num_results=20):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    urls = []
    start = 0
    while len(urls) < num_results:
        response = requests.get(
            'https://www.google.pl/search',
            params={'q': query, 'hl': 'pl', 'start': start},
            headers=headers
        )
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.select('div.yuRUbf a')
        for result in results:
            url = result['href']
            if url not in urls and is_polish_page(url, headers):
                urls.append(url)
            if len(urls) >= num_results:
                break
        start += 10
        time.sleep(random.uniform(1, 3))
        if not results:
            break
    return urls


def is_polish(text):
    polish_words = set(["jest", "się", "nie", "w", "na", "i", "z", "że", "to", "co"])
    words = re.findall(r'\b\w+\b', text.lower())
    polish_count = sum(1 for word in words if word in polish_words)
    return polish_count / len(words) > 0.1  # jeśli więcej niż 10% słów to polskie słowa kluczowe, uznajemy stronę za polską

def clean_text(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(' +', ' ', text)
    return text

def normalize_polish_chars(text):
    text = unicodedata.normalize('NFKC', text)
    return text

def fetch_page_content(url):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = session.get(url, headers=headers, timeout=10, verify=False)  # wyłączamy weryfikację certyfikatu
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['header', 'footer', 'script', 'style']):
            tag.decompose()
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        lists = [li.get_text() for li in soup.find_all('li')]
        h1 = [h.get_text() for h in soup.find_all('h1')]
        h2_h3 = [h.get_text() for h in soup.find_all(['h2', 'h3'])]
        content = paragraphs + lists + h1 + h2_h3
        text = ' '.join(content)
        text = clean_text(text)
        text = normalize_polish_chars(text).lower()
        if not is_polish(text):
            return None, None, None, None
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count < 100:
            return None, None, None, None
        return content, text, h1, h2_h3
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None, None, None

def extract_ngrams(text, n):
    clean_text = re.sub(r'[.,()–\-\[\]{}!"#$%&\'*+/;<=>?@\\^_`|~]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    words = clean_text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

def contains_only_stopwords(ngram, stopwords):
    words = ngram.split()
    return all(word in stopwords for word in words)

def process_texts(texts, headers_h1, headers_h2_h3):
    all_phrases = Counter()
    phrase_occurrences = defaultdict(int)
    all_ngrams_2 = Counter()
    all_ngrams_3 = Counter()
    all_ngrams_4 = Counter()
    ngram_2_occurrences = defaultdict(int)
    ngram_3_occurrences = defaultdict(int)
    ngram_4_occurrences = defaultdict(int)
    all_phrases_h1 = Counter()
    phrase_occurrences_h1 = defaultdict(int)
    all_ngrams_1_h1 = Counter()
    all_ngrams_2_h1 = Counter()
    all_ngrams_3_h1 = Counter()
    all_ngrams_4_h1 = Counter()
    ngram_1_occurrences_h1 = defaultdict(int)
    ngram_2_occurrences_h1 = defaultdict(int)
    ngram_3_occurrences_h1 = defaultdict(int)
    ngram_4_occurrences_h1 = defaultdict(int)
    all_phrases_h2_h3 = Counter()
    phrase_occurrences_h2_h3 = defaultdict(int)
    all_ngrams_2_h2_h3 = Counter()
    all_ngrams_3_h2_h3 = Counter()
    all_ngrams_4_h2_h3 = Counter()
    ngram_2_occurrences_h2_h3 = defaultdict(int)
    ngram_3_occurrences_h2_h3 = defaultdict(int)
    ngram_4_occurrences_h2_h3 = defaultdict(int)

    for text, h1, h2_h3 in zip(texts, headers_h1, headers_h2_h3):
        words = re.findall(r'\b\w+\b', text.lower())
        phrases = set(words) - polish_stopwords
        ngrams_2 = set(extract_ngrams(text, 2))
        ngrams_3 = set(extract_ngrams(text, 3))
        ngrams_4 = set(extract_ngrams(text, 4))

        all_phrases.update(phrases)
        all_ngrams_2.update([ngram for ngram in ngrams_2 if not contains_only_stopwords(ngram, polish_stopwords)])
        all_ngrams_3.update([ngram for ngram in ngrams_3 if not contains_only_stopwords(ngram, polish_stopwords)])
        all_ngrams_4.update([ngram for ngram in ngrams_4 if not contains_only_stopwords(ngram, polish_stopwords)])

        for phrase in phrases:
            phrase_occurrences[phrase] += 1
        for ngram in ngrams_2:
            if not contains_only_stopwords(ngram, polish_stopwords):
                ngram_2_occurrences[ngram] += 1
        for ngram in ngrams_3:
            if not contains_only_stopwords(ngram, polish_stopwords):
                ngram_3_occurrences[ngram] += 1
        for ngram in ngrams_4:
            if not contains_only_stopwords(ngram, polish_stopwords):
                ngram_4_occurrences[ngram] += 1

        for header in h1:
            header = normalize_polish_chars(header).lower()
            words_h1 = re.findall(r'\b\w+\b', header)
            phrases_h1 = set(words_h1) - polish_stopwords
            ngrams_1_h1 = set(words_h1)
            ngrams_2_h1 = set(extract_ngrams(header, 2))
            ngrams_3_h1 = set(extract_ngrams(header, 3))
            ngrams_4_h1 = set(extract_ngrams(header, 4))

            all_phrases_h1.update(phrases_h1)
            all_ngrams_1_h1.update([ngram for ngram in ngrams_1_h1 if not contains_only_stopwords(ngram, polish_stopwords)])
            all_ngrams_2_h1.update([ngram for ngram in ngrams_2_h1 if not contains_only_stopwords(ngram, polish_stopwords)])
            all_ngrams_3_h1.update([ngram for ngram in ngrams_3_h1 if not contains_only_stopwords(ngram, polish_stopwords)])
            all_ngrams_4_h1.update([ngram for ngram in ngrams_4_h1 if not contains_only_stopwords(ngram, polish_stopwords)])

            for phrase in phrases_h1:
                phrase_occurrences_h1[phrase] += 1
            for ngram in ngrams_1_h1:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_1_occurrences_h1[ngram] += 1
            for ngram in ngrams_2_h1:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_2_occurrences_h1[ngram] += 1
            for ngram in ngrams_3_h1:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_3_occurrences_h1[ngram] += 1
            for ngram in ngrams_4_h1:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_4_occurrences_h1[ngram] += 1

        for header in h2_h3:
            header = normalize_polish_chars(header).lower()
            words_h2_h3 = re.findall(r'\b\w+\b', header)
            phrases_h2_h3 = set(words_h2_h3) - polish_stopwords
            ngrams_2_h2_h3 = set(extract_ngrams(header, 2))
            ngrams_3_h2_h3 = set(extract_ngrams(header, 3))
            ngrams_4_h2_h3 = set(extract_ngrams(header, 4))

            all_phrases_h2_h3.update(phrases_h2_h3)
            all_ngrams_2_h2_h3.update([ngram for ngram in ngrams_2_h2_h3 if not contains_only_stopwords(ngram, polish_stopwords)])
            all_ngrams_3_h2_h3.update([ngram for ngram in ngrams_3_h2_h3 if not contains_only_stopwords(ngram, polish_stopwords)])
            all_ngrams_4_h2_h3.update([ngram for ngram in ngrams_4_h2_h3 if not contains_only_stopwords(ngram, polish_stopwords)])

            for phrase in phrases_h2_h3:
                phrase_occurrences_h2_h3[phrase] += 1
            for ngram in ngrams_2_h2_h3:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_2_occurrences_h2_h3[ngram] += 1
            for ngram in ngrams_3_h2_h3:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_3_occurrences_h2_h3[ngram] += 1
            for ngram in ngrams_4_h2_h3:
                if not contains_only_stopwords(ngram, polish_stopwords):
                    ngram_4_occurrences_h2_h3[ngram] += 1

    return (all_phrases, phrase_occurrences, all_ngrams_2, ngram_2_occurrences, all_ngrams_3, ngram_3_occurrences, all_ngrams_4, ngram_4_occurrences,
            all_phrases_h1, phrase_occurrences_h1, all_ngrams_1_h1, ngram_1_occurrences_h1, all_ngrams_2_h1, ngram_2_occurrences_h1, all_ngrams_3_h1, ngram_3_occurrences_h1, all_ngrams_4_h1, ngram_4_occurrences_h1,
            all_phrases_h2_h3, phrase_occurrences_h2_h3, all_ngrams_2_h2_h3, ngram_2_occurrences_h2_h3, all_ngrams_3_h2_h3, ngram_3_occurrences_h2_h3, all_ngrams_4_h2_h3, ngram_4_occurrences_h2_h3)

def landing_page(request):
    return render(request, 'blog/landing_page.html')

@csrf_exempt
def submit_landing_page(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        urls = search_google(query, num_results=20)
        texts, headers_h1, headers_h2_h3 = [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(fetch_page_content, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    content, text, h1, h2_h3 = future.result()
                    if text:
                        texts.append(text)
                        headers_h1.append(h1)
                        headers_h2_h3.append(h2_h3)
                except Exception as e:
                    print(f"Error processing result: {e}")

        (all_phrases, phrase_occurrences, all_ngrams_2, ngram_2_occurrences, all_ngrams_3, ngram_3_occurrences, all_ngrams_4, ngram_4_occurrences,
         all_phrases_h1, phrase_occurrences_h1, all_ngrams_1_h1, ngram_1_occurrences_h1, all_ngrams_2_h1, ngram_2_occurrences_h1, all_ngrams_3_h1, ngram_3_occurrences_h1, all_ngrams_4_h1, ngram_4_occurrences_h1,
         all_phrases_h2_h3, phrase_occurrences_h2_h3, all_ngrams_2_h2_h3, ngram_2_occurrences_h2_h3, all_ngrams_3_h2_h3, ngram_3_occurrences_h2_h3, all_ngrams_4_h2_h3, ngram_4_occurrences_h2_h3) = process_texts(texts, headers_h1, headers_h2_h3)

        num_pages = len(texts)
        half_documents = num_pages / 2
        phrases_50_percent = {phrase: count for phrase, count in all_phrases.items() if phrase_occurrences[phrase] >= half_documents}
        ngrams_2_20_percent = {ngram: count for ngram, count in all_ngrams_2.items() if ngram_2_occurrences[ngram] >= 0.2 * num_pages}
        ngrams_3_20_percent = {ngram: count for ngram, count in all_ngrams_3.items() if ngram_3_occurrences[ngram] >= 0.2 * num_pages}
        ngrams_4_20_percent = {ngram: count for ngram, count in all_ngrams_4.items() if ngram_4_occurrences[ngram] >= 0.2 * num_pages}
        phrases_50_percent_h1 = {phrase: count for phrase, count in all_phrases_h1.items() if phrase_occurrences_h1[phrase] >= half_documents}
        ngrams_1_20_percent_h1 = {ngram: count for ngram, count in all_ngrams_1_h1.items() if ngram_1_occurrences_h1[ngram] >= 0.2 * num_pages}
        ngrams_2_20_percent_h1 = {ngram: count for ngram, count in all_ngrams_2_h1.items() if ngram_2_occurrences_h1[ngram] >= 0.2 * num_pages}
        ngrams_3_20_percent_h1 = {ngram: count for ngram, count in all_ngrams_3_h1.items() if ngram_3_occurrences_h1[ngram] >= 0.2 * num_pages}
        ngrams_4_20_percent_h1 = {ngram: count for ngram, count in all_ngrams_4_h1.items() if ngram_4_occurrences_h1[ngram] >= 0.2 * num_pages}
        phrases_50_percent_h2_h3 = {phrase: count for phrase, count in all_phrases_h2_h3.items() if phrase_occurrences_h2_h3[phrase] >= half_documents}
        ngrams_2_20_percent_h2_h3 = {ngram: count for ngram, count in all_ngrams_2_h2_h3.items() if ngram_2_occurrences_h2_h3[ngram] >= 0.2 * num_pages}
        ngrams_3_20_percent_h2_h3 = {ngram: count for ngram, count in all_ngrams_3_h2_h3.items() if ngram_3_occurrences_h2_h3[ngram] >= 0.2 * num_pages}
        ngrams_4_20_percent_h2_h3 = {ngram: count for ngram, count in all_ngrams_4_h2_h3.items() if ngram_4_occurrences_h2_h3[ngram] >= 0.2 * num_pages}

        # Top 20 elements for each category or fewer if not enough data
        top_20_phrases = dict(Counter(phrases_50_percent).most_common(20))
        top_20_ngrams_2 = dict(Counter(ngrams_2_20_percent).most_common(20))
        top_20_ngrams_3 = dict(Counter(ngrams_3_20_percent).most_common(20))
        top_20_ngrams_4 = dict(Counter(ngrams_4_20_percent).most_common(20))
        top_20_phrases_h1 = dict(Counter(phrases_50_percent_h1).most_common(20))
        top_20_ngrams_1_h1 = dict(Counter(ngrams_1_20_percent_h1).most_common(20))
        top_20_ngrams_2_h1 = dict(Counter(ngrams_2_20_percent_h1).most_common(20))
        top_20_ngrams_3_h1 = dict(Counter(ngrams_3_20_percent_h1).most_common(20))
        top_20_ngrams_4_h1 = dict(Counter(ngrams_4_20_percent_h1).most_common(20))
        top_20_phrases_h2_h3 = dict(Counter(phrases_50_percent_h2_h3).most_common(20))
        top_20_ngrams_2_h2_h3 = dict(Counter(ngrams_2_20_percent_h2_h3).most_common(20))
        top_20_ngrams_3_h2_h3 = dict(Counter(ngrams_3_20_percent_h2_h3).most_common(20))
        top_20_ngrams_4_h2_h3 = dict(Counter(ngrams_4_20_percent_h2_h3).most_common(20))

        # Prepare files to download
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        for data, file_name in [
            (phrases_50_percent, 'phrases_50_percent.xlsx'),
            (ngrams_2_20_percent, 'ngrams_2_20_percent.xlsx'),
            (ngrams_3_20_percent, 'ngrams_3_20_percent.xlsx'),
            (ngrams_4_20_percent, 'ngrams_4_20_percent.xlsx'),
            (phrases_50_percent_h1, 'phrases_50_percent_h1.xlsx'),
            (ngrams_1_20_percent_h1, 'ngrams_1_20_percent_h1.xlsx'),
            (ngrams_2_20_percent_h1, 'ngrams_2_20_percent_h1.xlsx'),
            (ngrams_3_20_percent_h1, 'ngrams_3_20_percent_h1.xlsx'),
            (ngrams_4_20_percent_h1, 'ngrams_4_20_percent_h1.xlsx'),
            (phrases_50_percent_h2_h3, 'phrases_50_percent_h2_h3.xlsx'),
            (ngrams_2_20_percent_h2_h3, 'ngrams_2_20_percent_h2_h3.xlsx'),
            (ngrams_3_20_percent_h2_h3, 'ngrams_3_20_percent_h2_h3.xlsx'),
            (ngrams_4_20_percent_h2_h3, 'ngrams_4_20_percent_h2_h3.xlsx')
        ]:
            if data:
                file_path = os.path.join(temp_dir, file_name)
                df = pd.DataFrame.from_dict(data, orient='index')
                df.to_excel(file_path)
                file_paths.append(file_path)

        # Generate ZIP file
        zip_subdir = "results"
        zip_filename = "{}.zip".format(zip_subdir)
        s = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)

        with zipfile.ZipFile(s, "w") as zf:
            for fpath in file_paths:
                fdir, fname = os.path.split(fpath)
                zip_path = os.path.join(zip_subdir, fname)
                zf.write(fpath, zip_path)

        shutil.rmtree(temp_dir)

        # Save the zip file path to be available for download
        request.session['zip_file_path'] = s.name

        context = {
            'query': query,
            'top_20_phrases': top_20_phrases if top_20_phrases else None,
            'top_20_ngrams_2': top_20_ngrams_2 if top_20_ngrams_2 else None,
            'top_20_ngrams_3': top_20_ngrams_3 if top_20_ngrams_3 else None,
            'top_20_ngrams_4': top_20_ngrams_4 if top_20_ngrams_4 else None,
            'top_20_phrases_h1': top_20_phrases_h1 if top_20_phrases_h1 else None,
            'top_20_ngrams_1_h1': top_20_ngrams_1_h1 if top_20_ngrams_1_h1 else None,
            'top_20_ngrams_2_h1': top_20_ngrams_2_h1 if top_20_ngrams_2_h1 else None,
            'top_20_ngrams_3_h1': top_20_ngrams_3_h1 if top_20_ngrams_3_h1 else None,
            'top_20_ngrams_4_h1': top_20_ngrams_4_h1 if top_20_ngrams_4_h1 else None,
            'top_20_phrases_h2_h3': top_20_phrases_h2_h3 if top_20_phrases_h2_h3 else None,
            'top_20_ngrams_2_h2_h3': top_20_ngrams_2_h2_h3 if top_20_ngrams_2_h2_h3 else None,
            'top_20_ngrams_3_h2_h3': top_20_ngrams_3_h2_h3 if top_20_ngrams_3_h2_h3 else None,
            'top_20_ngrams_4_h2_h3': top_20_ngrams_4_h2_h3 if top_20_ngrams_4_h2_h3 else None,
            'download_available': True
        }

        return render(request, 'blog/landing_page.html', context)

    return render(request, 'blog/landing_page.html', {'download_available': False})



def download_zip(request):
    zip_file_path = request.session.get('zip_file_path')
    if zip_file_path and os.path.exists(zip_file_path):
        with open(zip_file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type="application/zip")
            response['Content-Disposition'] = f'attachment; filename=results.zip'
        os.remove(zip_file_path)
        del request.session['zip_file_path']
        return response
    return HttpResponse("No file available for download.")
