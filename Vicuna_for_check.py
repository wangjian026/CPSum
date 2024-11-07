import utils
from largemodel import vicunaUsercase


def sup_opp_rule(answer, document, sentence):
    answer = answer.replace(document, '##########doc#######').replace(sentence, '#######sen#########')
    support_words = set(utils.readsettings('support_words'))
    against_words = set(utils.readsettings('against_words'))
    notclaarly_words = set(utils.readsettings('notclaarly_words'))

    if any(keyword in answer.lower() for keyword in notclaarly_words):
        predicted_class = "not clearly"
    elif any(keyword in answer.lower() for keyword in against_words):
        predicted_class = "against"
    elif any(keyword in answer.lower() for keyword in support_words) and any(
            keyword in answer.lower() for keyword in against_words):
        predicted_class = "not clearly"
    else:
        predicted_class = "support"
    return predicted_class


def process_element(client_socket_eva,data):

    document = data[0]
    sentence = data[1]
    prompt = utils.readsettings('sup_opp_prompt')
    prompt = prompt.replace('{d}', document).replace('{s}', sentence)
    exit_code = '!!reset'
    vicunaUsercase.getAnswer(client_socket_eva, exit_code)
    answer = vicunaUsercase.getAnswer(client_socket=client_socket_eva, query=prompt)
    answer = sup_opp_rule(answer, document, sentence)
    if answer == 'support':
        return 1
    elif answer == 'against':
        return -1
    elif answer == 'not clearly':
        return 0

def construct_sup_opp_set_Vicuna(clients_list,Documents, sentence):
    result = {}
    sup_set = set()
    opp_set = set()
    neu_set = set()
    data_list = []
    for doucment in Documents:
        data_list.append([doucment, sentence])

    # Specify the client to be used for evaluation
    check_result = process_element(client_socket_eva=clients_list[1],data = data_list)
    sup_set.update([idx for idx, val in enumerate(check_result) if val == 1])
    opp_set.update([idx for idx, val in enumerate(check_result) if val == -1])
    neu_set.update([idx for idx, val in enumerate(check_result) if val == 0])

    result['sup_set'] = sup_set
    result['opp_set'] = opp_set
    result['neu_set'] = neu_set

    return result