import os
import re
import json
import socket

from nltk import word_tokenize,pos_tag
from nltk.corpus import stopwords
import string
import nltk

stop_words = set(stopwords.words('english'))

def jaccard_coefficient(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0

    return intersection / union

def extract_nouns(text):
    words = word_tokenize(text.lower())
    tagged_words = pos_tag(words)
    nouns = [word for word, pos in tagged_words if pos.startswith('N')]
    return nouns


def creatClient(host, port):
    # Creating a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #
    host = host
    # host = socket.gethostname()
    port = port
    try:
        client_socket.connect((host, port))
        print("Successfully connected to the server. port:", port)
    except Exception as e:
        print(f"Failed to connect to the server {port}. Error: {e}")
    return client_socket
def split_sentence_with_conjunction(sentence):
    # Define the list of transitions
    conjunctions = ["but","with some", "however", "although", "though", "nevertheless", "yet","while", "on the other hand", "in contrast", "whereas", "conversely", "even though","on the contrary"]

    # Finding the location of punctuation marks
    punctuation_positions = [pos for pos, char in enumerate(sentence) if char in [',', ';', '.']]
    print(punctuation_positions)
    for pos in reversed(punctuation_positions):
        # Intercept after punctuation
        part1 = sentence[:pos+1].strip()
        part2 = sentence[pos+1:].strip()

        # Determine if the second part begins with a turn of phrase
        for conjunction in conjunctions:
            if part2.lower().startswith(conjunction):
                return [part1, part2]


    return [sentence]


def split_into_sentences(text):
    """
    :param text:
    :return:
    """
    sentences = nltk.sent_tokenize(text)
    return sentences
def readsettings(key):

    f = open('settings.json', 'r')
    settings = json.load(f)
    return settings[key]

def readtext(path):
    with open(path, encoding='utf-8') as file_obj:
        contents = file_obj.readlines()
    return contents


def remove_surrogate_pairs(input_string):
    # 使用正则表达式匹配代理对
    surrogate_pattern = re.compile('[\ud800-\udbff][\udc00-\udfff]')

    # 使用 sub 方法将代理对替换为空字符串
    cleaned_string = surrogate_pattern.sub(r'', input_string)

    return cleaned_string

