import re

def main():
    sentence = input()
    BOW_dict, BOW = create_BOW(sentence)

    print(BOW_dict)
    print(BOW)

def create_BOW(sentence):
    # Exercise
    list = []
    lower_sentence = sentence.lower()
    convert_sentence = replace_non_alphabetic_chars_to_space(lower_sentence)
    
    list = convert_sentence.split(' ')

    for item in list:

        if len(item) == 0:

            list.remove (item)

    #print (list)
    #print (len(list))
    bow_dict = {}
    i = 0

    for item in list:

        if item not in bow_dict:
            bow_dict[item] = i
            i = i + 1

    #print (bow_dict)
    #print (len(bow_dict))
    #print (sentence)

    bow = [0] * len(bow_dict)
    #print (bow)
    
    for item in list:

        bow[bow_dict.get (item)] = bow[bow_dict.get (item)] + 1
    
    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

if __name__ == "__main__":
    main()