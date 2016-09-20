import re
import math

def main():
    # 1
    training_sentence = input()
    training_model = create_BOW(training_sentence)
    #print (training_model[0])
    #print (training_model[1])
    #print (len (training_model[0]))
    #print (len (training_model[1]))
    #print (sum (training_model[1]))
    #print (training_model[0])

    # 2
    testing_sentence = input()
    testing_model = create_BOW(testing_sentence)
    #print (testing_model[0])
    #print (testing_model[1])
    #print (len (testing_model[0]))
    #print (len (testing_model[1]))
    #print (sum (testing_model[1]))

    # 3
    alpha = float(input())

    print(calculate_doc_prob(training_model, testing_model, alpha))

def calculate_doc_prob(training_model, testing_model, alpha):
    # Implement likelihood function here...
    logprob = 0 
    
    for item in testing_model[0]:

        #print (training_model[1][training_model[0][item]])
        #print (item)

        word_freq = testing_model[1][testing_model[0][item]]
        #print (word_freq)

        for i in range(0, word_freq):
            
            try:
                
                logprob += math.log (training_model[1][training_model[0][item]] + alpha )
                logprob -= math.log (sum (training_model[1]) + alpha * (len (training_model[0])))
            #print (training_model[0][item] + alpha)
            #print (sum (training_model[1]) + alpha * (len (training_model[0])))

            except:

                logprob += math.log (alpha)
                logprob -= math.log (sum (training_model[1]) + alpha * (len (training_model[0])))
            #print (alpha)
            #print (sum (training_model[1]) + alpha * (len (training_model[0])))

    return logprob

#def calculate_doc_prob(training_model, testing_model, alpha):
#    logprob = 0

#    num_tokens_training = sum(training_model[1])
#    num_words_training = len(training_model[0])

#    for word in testing_model[0]:
#        word_freq = testing_model[1][testing_model[0][word]]
#        word_freq_in_training = 0
#        if word in training_model[0]:
#            word_freq_in_training = training_model[1][training_model[0][word]]
#        for i in range(0, word_freq):
#            logprob += math.log(word_freq_in_training + alpha)
#            logprob -= math.log(num_tokens_training + num_words_training * alpha)
#            print (word_freq_in_training + alpha)
#            print (num_tokens_training + num_words_training * alpha)

#    return logprob

def create_BOW(sentence):
    bow_dict = {}
    bow = []

    sentence = sentence.lower()
    sentence = replace_non_alphabetic_chars_to_space(sentence)
    words = sentence.split(' ')
    for token in words:
        if len(token) < 1: continue
        if token not in bow_dict:
            new_idx = len(bow)
            bow.append(0)
            bow_dict[token] = new_idx
        bow[bow_dict[token]] += 1

    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

if __name__ == "__main__":
    main()
