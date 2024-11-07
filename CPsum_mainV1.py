import os

import nltk
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

from nltk.tree import Tree
from nltk.corpus import stopwords
from collections import Counter

from Vicuna_for_check import construct_sup_opp_set_Vicuna
from largemodel import vicunaUsercase
import utils

print(nltk.data.path)
print(stopwords)
stop_words = set(stopwords.words('english'))
custom_stopwords = {"not", "no", "isn't", '%', 'run', 'years'}
mode = utils.readsettings('use_mode')
import stanza
tokenizer = AutoTokenizer.from_pretrained("./largemodel/roberta-argument")
model = RobertaForSequenceClassification.from_pretrained("./largemodel/roberta-argument")
# Loading the English Dependency Syntax Model (English Model)
stanza.download('en')  #
stanza_nlp = stanza.Pipeline('en', download_method=None)  #


def preprocess(text):
    new_text = []
    text = utils.remove_surrogate_pairs(text)
    for t in text.split(" "):
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = t.replace('URL_LINK', '')
        t = '' if t.startswith('#') and len(t) > 1 else t
        new_text.append(t)
    return " ".join(new_text)

def process_opinion(opinion_path):
    document_file_list = os.listdir(opinion_path)
    opinion_list = []
    for index, item in enumerate(document_file_list):
        temp_path = os.path.join(opinion_path, item)
        text_content = eval(utils.readtext(temp_path)[0])
        opinion_list.append(text_content)
    return opinion_list
def getData(dir):
    # 获取不同观点
    opinion_lists = process_opinion(dir)
    cluster = []
    for item in opinion_lists:
        data = {}
        majority_opinion = ''
        minority_opinion = ''
        Documents = []
        if 'tweets' in item.keys():
            Documents = [preprocess(tweet).strip() for tweet in item['tweets']]
        if 'majority_opinion' in item.keys():
            majority_opinion = item['majority_opinion']
        if 'minority_opinions' in item.keys():
            minority_opinion = item['minority_opinions']
        if 'main_story' in item.keys():
            main_story = item['main_story']

        data['Documents'] = Documents
        data['majority_opinion'] = majority_opinion
        data['minority_opinion'] = minority_opinion
        data['main_story'] = main_story
        cluster.append(data)
    return cluster

def get_continuous_chunks_by_depv2_pos_multid(Documents, unwant_words, chunk_func=nltk.ne_chunk):
    """
    Use stanford's tool stanza to get the dependency tree, then get the label of each word, then based on the rules, extract the nouns (there will be inconsistencies
    between the labels of the words in the dependency tree and the labels of the words in the get_continuous_chunks_by_nltk_pos method)
    :param Documents:
    :param chunk_func:
    :unwant_words: ['%']
    :return: continuous_chunks_list,  Each item is a list of noun blocks in the text [['social distancing', 'small bathrooms', 'menus', 'masks', 'people', 'air'],[]]
    """
    in_docs = [stanza.Document([], text=d) for d in Documents]  # Wrap each document with a stanza.Document object
    out_docs = stanza_nlp(in_docs)  # Call the neural pipeline on this list of Documents
    continuous_chunks_list = []
    for doc in out_docs:
        leaves_with_labels = []

        for sentence in doc.sentences:
            for word in sentence.words:
                if word.text.lower() not in stop_words or word.text.lower() in custom_stopwords:
                    leaves_with_label = (word.text, word.xpos)
                    leaves_with_labels.append(leaves_with_label)
        chunked = chunk_func(leaves_with_labels)

        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if type(subtree) == Tree:
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
                # print(current_chunk)
            if current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        continuous_chunks_list.append(continuous_chunk)
    return continuous_chunks_list, out_docs

def getTopic(Documents, h=1):
    # 返回每个document的名词块列表，及依存句法树
    NP = "NP: {(<NN\w?>|<JJ>*)+.*<NN\w?>}"
    # NP = "NP: {(<JJ>*)+.*<NN\w?>}"
    NP = r"""NP: {(<NN\w?>|<JJ>*)+.*<NN\w?>} {<VB.*><RB.*>?}"""
    grammar = r"""
         NP: {(<NN\w?>|<JJ>*)+.*<NN\w?>}   # noun phrase
         VB_RB: {<VB><RB>}     # Verbs + Adverbs
     """
    unwant_words = ['%']
    chunker = nltk.RegexpParser(grammar)
    continuous_chunks_list, out_docs = get_continuous_chunks_by_depv2_pos_multid(Documents, unwant_words, chunker.parse)
    # print('continuous_chunks_list:',continuous_chunks_list)
    # out_docs[0].sentences[0].print_dependencies()

    # Count the words with the highest word frequency to get the focus object
    combined_list = []
    for temp in continuous_chunks_list:
        combined_list = combined_list + temp
    element_count = Counter(combined_list)
    most_common_elements = element_count.most_common(3)
    print('most_common_elements:', most_common_elements)
    print('entity_word:', most_common_elements[0][0])
    print('entity_word', most_common_elements[1][0])
    print('entity_word', most_common_elements[2][0])
    entity_word = most_common_elements[0][0]
    return str(entity_word)


def create_sum_prompt(Documents,selected_sentences, entity_word, FLAG):
    """
    :param Documents:
    :param selected_sentences: the key opinion reference
    :param entity_word: topic
    :param FLAG:
    :return:
    """

    entity_word = str(entity_word)
    Documents = str(Documents)
    if FLAG == 0:
        summary_prompt = str(utils.readsettings('initial_summary_prompt_bak6')).replace('#{topic}',
                                                                                      entity_word)
    else:
        summary_prompt = str(utils.readsettings('summary_prompt_bak6')).replace('#{topic}', entity_word)
        summary_prompt = summary_prompt.replace('#{R}', str(selected_sentences))

    summary_prompt = summary_prompt.replace('#{document}', str(Documents))
    return summary_prompt

def is_argument(inputs):
    inputs = tokenizer(
        inputs,
        padding=True,
        return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    max_index_per_row = torch.argmax(logits, dim=1)
    predicted_class_ids = max_index_per_row.tolist()
    is_argument = []
    for i in range(len(predicted_class_ids)):
        if model.config.id2label[predicted_class_ids[i]] == 'ARGUMENT':
            is_argument.append(True)
        else:
            is_argument.append(False)
    return is_argument
def fliter_sententce_byargument(sentences):
    """
    :param sentences:
    :return: Returns sentences containing only opinions
    """
    arguments = is_argument(sentences)
    result = [sen for index, sen in enumerate(sentences) if arguments[index] == True]
    return result
def process_summay_for_calreward(summary):
    """

    :param summary:
    :return: Cutting summaries into short sentences
    """
    first_sentences = utils.split_into_sentences(summary)

    # For each sentence, if there is a turn of phrase, slice it again
    sentences = []
    for sentence in first_sentences:
        sentences.extend(utils.split_sentence_with_conjunction(sentence))
    global mode
    # Remove sentences that do not contain an argument.
    sentences = fliter_sententce_byargument(sentences)
    return sentences


def cal_Div_score(source,target,sents_info):
    source_supset = sents_info[source]['sup_set']
    target_supset = sents_info[target]['sup_set']
    value = 1-utils.jaccard_coefficient(source_supset, target_supset)

    return value

def average_2d_array(arr):
    total_sum = 0
    count = 0

    for row in arr:
        for value in row:
            total_sum += value
            count += 1

    if count == 0:
        return 0
    else:
        return total_sum / count


def cal_reward(client_socket, tagart_sentences, Documents):
    """

    :param client_socket:
    :param tagart_sentences: Target Sentence List
    :param Documents:
    :return: div_score, cov_score, sents_info
    """
    sents_info = {}  # Save the support set, opposition set, and neutrality set for each sentence
    mode = utils.readsettings('use_mode')
    sup_sets=set()
    #Get the support set for each sentence and also get the set of support sets
    for idx, sentence in enumerate(tagart_sentences):
        print(f"Sentence {idx}: {sentence}")

        sent_result = construct_sup_opp_set_Vicuna(clients_list=client_socket, Documents=Documents, sentence=sentence)

        sents_info[sentence] = sent_result  # Combine the results of each sentence into a dictionary
        # with the key being the sentence and the value being the various attributes of that sentence
        sup_sets = sup_sets.union(sent_result['sup_set'])

    #Calculate the support score based on the support set
    cov_score = len(sup_sets)/len(Documents)

    #Access to Diversity Score
    array = [[0 for _ in range(len(tagart_sentences))] for _ in range(len(tagart_sentences))]
    for idx, source in enumerate(tagart_sentences):
        for idx2, target in enumerate(tagart_sentences):
            if idx2==idx:
                array[idx][idx2] = 0
            if idx2>idx:
                value = cal_Div_score(source,target,sents_info)
                array[idx][idx2] = value
                array[idx2][idx] = value
    div_score = average_2d_array(array)
    print('summary of cov_score and div_score:',cov_score,div_score)

    return div_score, cov_score, sents_info


def total_reward(div_score, cov_score):
    coverage = utils.readsettings('coverage')
    diversity = utils.readsettings('diversity')
    reward = coverage * cov_score + diversity * div_score
    return reward

def compute_diversity(sentence, selected_sentences, union_sent):
    """
    Calculate the diversity score between a sentence and the set of selected sentences
    :param sentence: target sentence
    :param selected_sentences: the selected sentences set
    :param union_sent: All information about the candidate sentence，union_sent = {'sentence':{'sup_set':sup_set, '':''}}
    :return: diversity score，float
    """
    values = []
    for s in selected_sentences:
        value = utils.jaccard_coefficient(union_sent[sentence]['sup_set'], union_sent[s]['sup_set'])
        values.append(value)
    return sum(values) / len(values)
    # return (utils.jaccard_coefficient(union_sent[sentence]['sup_set'],union_sent[s]['sup_set']) for s in selected_sentences)
    #
    # return sum(utils.jaccard_coefficient(union_sent[sentence]['sup_set'],union_sent[s]['sup_set']) for s in selected_sentences)


def stopping_select_condition_bydiversity(max_diversity_sentence):
    """
    :param max_diversity_sentence: diversity score
    :return:
    """
    # diversity score  threshold

    beta = utils.readsettings('beta')
    if max_diversity_sentence < beta:
        return True
    else:
        return False



def greedy_selection(candicate_sents, Documents):
    """
    From the given set, the sentences that satisfy the conditions are selected based on support and difference to form the key opinion reference     :param candicate_sents: 给定候选句子集合，candicate_sents = {'sentence':{'sup_set':sup_set, '':''}}
    :param Documents:
    :return:
    """
    selected_sentences = []
    remaining_sentences = set(candicate_sents.keys())

    # 计算支持度
    support_scores = [(sentence, len(candicate_sents[sentence]['sup_set']) / len(Documents)) for sentence in
                      remaining_sentences]
    logs = {}

    # Filter out sentences with lower support
    temp = remaining_sentences.copy()
    for sentence in temp:
        supportscore = [data[1] for data in support_scores if data[0] == sentence]
        if supportscore[0] < utils.readsettings('min_support_ratio'):
            try:
                remaining_sentences.remove(sentence)  #
            except KeyError:
                print("Element not found in the set.")
    if len(remaining_sentences) == 0:
        return selected_sentences, logs
    else:
        max_support_sentence, max_suppport_score = max(support_scores, key=lambda x: x[1])
        logs['max_suppport_score'] = max_suppport_score
        selected_sentences.append(max_support_sentence)
        remaining_sentences.remove(max_support_sentence)

    max_diversity_scores = []
    while remaining_sentences:
        # select the sentence that has the highest variance from the selected sentence.
        diversity_scores = [(sentence, compute_diversity(sentence, selected_sentences, candicate_sents)) for sentence in
                            remaining_sentences]
        max_diversity_sentence, max_diversity_score = max(diversity_scores, key=lambda x: x[1])
        max_diversity_scores.append(max_diversity_score)
        # Stop condition, stop when maximum variability is less than beta

        if stopping_select_condition_bydiversity(max_diversity_score):
            break
        selected_sentences.append(max_diversity_sentence)
        remaining_sentences.remove(max_diversity_sentence)

    logs['max_diversity_scores'] = max_diversity_scores

    #
    return selected_sentences, logs

def update_R(now_sents_info, previous_selected_sentences_info, Documents):
    """
    update the key opinion reference
    :param now_sents_info: info of the current  iteration {'sentence':{'sup_set':sup_set, '':''}}
    :param previous_selected_sentences_info: info in the previous iteration {'sentence':{'sup_set':sup_set, '':''}}
    :param Documents:
    :return: the selected sentences list()；
    """
    # Mix the generated sentences with the newly generated sentences,
    candicate_sents = {**now_sents_info, **previous_selected_sentences_info}

    if len(candicate_sents) == 0:
        selected_sentences = []
        R_flag = 0
        return selected_sentences, R_flag, {}

    selected_sentences, logs = greedy_selection(candicate_sents=candicate_sents, Documents=Documents)
    # R_flag=0means not updated.
    previous_sentence_inR = previous_selected_sentences_info.keys()
    R_flag = 0 if bool(sorted(previous_sentence_inR) == sorted(selected_sentences)) else 1

    return selected_sentences, R_flag, logs


def Iteration_stopping_conditions(previous_R, R_flag, rewards):
    message = {}
    reward_threshold_convergence = utils.readsettings('reward_threshold_convergence')

    if len(rewards) > 1:
        if reward_threshold_convergence > abs(rewards[-1] - rewards[-2]):
            message['info'] = '104'
            message['status'] = False
            return message
    if previous_R + R_flag == 0:
        message['info'] = '102'
        message['status'] = False
    else:
        message['info'] = '103'
        message['status'] = True
    return message

def getSummary(client_sockets, Documents):
    summaries = [] # Saves the summaries produced by each iteration.
    key_opinion_references = [] # Saves the key opinion references produced by each iteration.
    rewards = [] # Saves the rewards of summary  for each iteration.
    sup_agg_info = []  # Saves the support set and opposing set of summary  for each iteration.

    # Information to be saved for the first iteration.
    key_opinion_reference_info={} #
    key_opinion_references.append([])
    sup_agg_info.append({})

    iter_num = 0  # the number of iteration


    # 1, prompt construction

    # 1.1 get the topic
    topic = getTopic(Documents)
    prompt = create_sum_prompt(Documents=Documents,selected_sentences=[],entity_word=topic,
                               FLAG=0)
    # 1.2 generate the initial  summary
    exit_code = '!!reset'  # Will close the server receiver, closing the session
    vicunaUsercase.getAnswer(client_socket_sum, query=exit_code)
    role = 'You\'re a summary generator.'
    vicunaUsercase.getAnswer(client_socket_sum, query=role)
    summary = vicunaUsercase.getAnswer(client_socket_sum, query=prompt)
    summaries.append(summary)

    previous_R = 1
    logss = []
    while True:
        iter_num += 1
        max_iterNum = utils.readsettings('max_iterNum')
        if iter_num > max_iterNum:
            break

        # 2, reward calculation
        # 2.1, cut the summary
        sentences_in_summary = process_summay_for_calreward(summary)

        # 2.2, Calculate the coverage(cov_score) and diversity scores (div_score) of the summaries,
        # as well as the support and opposition set information (thisstep_sents_info) for each sentence.
        div_score, cov_score, thisstep_sents_info = cal_reward(
            client_socket=client_sockets,
            tagart_sentences=sentences_in_summary,
            Documents=Documents)

        rewards.append(total_reward(div_score, cov_score))

        # 3 Prompt calibration
        # Get new key opiinion reference (key_opinion_reference;list);
        # R_flag is tht flag to show whether the previous key opinion reference is updated.
        key_opinion_reference, R_flag, logs = update_R(now_sents_info=thisstep_sents_info, previous_selected_sentences_info=key_opinion_reference_info,
                                   Documents=Documents)  # 句子列表

        # 3.1 temp save the sentence info for the next iteration
        candicate_sents = {**thisstep_sents_info, **key_opinion_reference_info}
        key_opinion_reference_info = {s: candicate_sents[s] for s in key_opinion_reference}  # 句子的所有信息

        # 3.2 Determine whether the stop condition is reached by reward and R_flag
        message = Iteration_stopping_conditions(previous_R, R_flag, rewards)
        previous_R = R_flag


        # Save information for final results analysis
        key_opinion_references.append(key_opinion_reference)
        logss.append(logs)
        sup_agg_info.append(thisstep_sents_info)

        # 4 summary generation
        if message['status']:
            # 形成新的prompt，和摘要
            exit_code = '!!reset'  # 会关闭服务器接收端，关闭会话
            vicunaUsercase.getAnswer(client_socket_sum, query=exit_code)
            role = 'You\'re a summary generator.'
            vicunaUsercase.getAnswer(client_socket_sum, query=role)

            prompt = create_sum_prompt(Documents=Documents,selected_sentences=key_opinion_reference,entity_word=topic,
                                                                       FLAG=1)
            summary = vicunaUsercase.getAnswer(client_socket_sum, query=prompt)
            summaries.append(summary)
        else:
            break
    with open('detail_results.txt', 'a+', encoding='utf-8') as file:
        # 写入内容
        details = {}
        details['topic'] = topic
        details['references'] = key_opinion_references
        details['sup_agg_info'] = sup_agg_info
        file.write(str(details) + '\n')
    with open('summary_iter.txt', 'a+', encoding='utf-8') as file:
        # 写入内容
        file.write(str(summaries) + '\n')
    return summary

if __name__ == '__main__':

    host = '127.0.0.1'
    sum_port = utils.readsettings('localport_sum')
    client_socket_sum = utils.creatClient(host, sum_port)


    testfile_dir = './Data/MOSdata/MOS_corpus_hydrated/testing/opi'
    cluster = getData(testfile_dir)
    for index, instance in enumerate(cluster):
        try:
            print(f'%%%%%%%%%%%%%%the {index} instance%%%%%%%%%%%%%%%%%')
            Documents = instance['Documents']
            Documents = [preprocess(tweet).strip() for tweet in Documents]
            summary = getSummary(client_sockets=[client_socket_sum],
                                 Documents=Documents)
        except Exception as ex:
            exit_code = '!!exit'  # 会关闭服务器接收端，关闭会话
            vicunaUsercase.getAnswer(client_socket_sum, query=exit_code)

            import traceback

            traceback.print_exc()