#Gramatical_database_rutine
# -*- coding: utf-8 -*-
'''Script to create databases with crandom vector corresponding to words in a sentence'''
#import libraries
import numpy as np
import cPickle

# Read a list of words from a textfie and store them in an array. I'll use 
# the shape of this array to create the lists of codes to feed the RNN
# the files have to use '   ' as delimiter
def array_from_file(filename):
    emptylist = []
    with open(filename) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            # if you need to use the file content as numbers:
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            emptylist.append(inner_list)
    f.close()
    return np.array(emptylist)
# Get arrays from
Gram_array = array_from_file('Grammatical_Clean.txt')
Jabb_array = array_from_file('Jabberwocky_Clean.txt')
OnlyNPs12_array = array_from_file('OnlyNPs_1_2_Clean.txt')
OnlyNPs13_array = array_from_file('OnlyNPs_1_3_Clean.txt')
OnlyNPs_array = array_from_file('OnlyNPs_Clean.txt')
Wordlist_array = array_from_file('WordList_Clean.txt') 
# Get shapes of the arrays
Gram_shape = Gram_array.shape
Jabb_shape = Jabb_array.shape
OnlyNPs12_shape = OnlyNPs12_array.shape
OnlyNPs13_shape = OnlyNPs13_array.shape
OnlyNPs_shape = OnlyNPs_array.shape
Wordlist_shape = Wordlist_array.shape
# Create codes for type of words in the grammatical list
adj = [0., 0., 1.]
noun = [0., 1., 0.]
verb = [1., 0., 0.]
# Create base patterns for every database
Gram_patt = np.array([adj, noun, verb, noun])
Jabb_patt = np.array([adj, noun, verb, noun])
OnlyNPs12_patt = np.array([adj, adj, noun])
OnlyNPs13_patt = np.array([adj, adj, adj, noun])
OnlyNPs_patt = np.array([adj, noun])
# Get modify shapes for broadcasting...
Gram_broadcast = (Gram_shape[0],Gram_shape[1], 3)
Jabb_broadcast = (Jabb_shape[0],Jabb_shape[1], 3)
OnlyNPs12_broadcast = (OnlyNPs12_shape[0],OnlyNPs12_shape[1], 3)
OnlyNPs13_broadcast = (OnlyNPs13_shape[0],OnlyNPs13_shape[1], 3)
OnlyNPs_broadcast = (OnlyNPs_shape[0],OnlyNPs_shape[1], 3)
# Broadcast base patterns to the corresponding database shape
Gram_typeword = np.broadcast_to(Gram_patt, Gram_broadcast)
Jabb_typeword = np.broadcast_to(Jabb_patt, Jabb_broadcast)
OnlyNPs12_typeword = np.broadcast_to(OnlyNPs12_patt, OnlyNPs12_broadcast)
OnlyNPs13_typeword = np.broadcast_to(OnlyNPs13_patt, OnlyNPs13_broadcast)
OnlyNPs_typeword = np.broadcast_to(OnlyNPs_patt, OnlyNPs_broadcast)

# Create array of non repeating random codes with the same shape as the
# database. Also, none of the codes should be equal to 0
def codearray(array_shape, n_vector):
    empytlist = []
    endcode = np.zeros(n_vector)
    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            a = np.random.randint(2, size=n_vector)
            while any((a == x).all() for x in empytlist) or any((a == endcode).all() for x in empytlist):
                a = np.random.randint(2, size=n_vector)
            empytlist.append(a)
    vector_array = np.array(empytlist).reshape(array_shape[0],array_shape[1], n_vector)
    return vector_array
    
Gram_code_only = codearray(Gram_shape,10)
Jabb_code_only = codearray(Jabb_shape,10)
OnlyNPs12_code_only = codearray(OnlyNPs12_shape,10)
OnlyNPs13_code_only = codearray(OnlyNPs13_shape,10)
OnlyNPs_code_only = codearray(OnlyNPs_shape,10)
Wordlist_code_only = codearray(Wordlist_shape,10)
                                              
##asign the same code to repeating words. For this compare all pairs in
##the the array of words and asign the same code in the code array if 
## the words are the same 
def match_rep_words(codearray, wordarray):
    copy = codearray
    for h in range(codearray.shape[0]):
        for i in range(codearray.shape[1]):
            for j in range(codearray.shape[0]):
                for k in range(codearray.shape[1]):
                    if (h != j) and (i !=k):
                        if wordarray[h,i] == wordarray[j,k]:
                            #print h,i,j,k
                            copy[j,k] = copy[h,i]
    return copy
Gram_code_only = match_rep_words(Gram_code_only, Gram_array)
Jabb_code_only = match_rep_words(Jabb_code_only, Jabb_array)
OnlyNPs12_code_only = match_rep_words(OnlyNPs12_code_only, OnlyNPs12_array)
OnlyNPs13_code_only = match_rep_words(OnlyNPs13_code_only, OnlyNPs13_array)
OnlyNPs_code_only = match_rep_words(OnlyNPs_code_only, OnlyNPs_array)
Wordlist_code_only = match_rep_words(Wordlist_code_only, Wordlist_array)

# Add type of word code to every entry of gram_code_only
Gram_code_typeword = np.concatenate((Gram_typeword, Gram_code_only), axis=2)
Jabb_code_typeword = np.concatenate((Jabb_typeword, Jabb_code_only), axis=2)
OnlyNPs12_code_typeword = np.concatenate((OnlyNPs12_typeword, OnlyNPs12_code_only), axis=2)

OnlyNPs13_code_typeword = np.concatenate((OnlyNPs13_typeword, OnlyNPs13_code_only), axis=2)

OnlyNPs_code_typeword = np.concatenate((OnlyNPs_typeword, OnlyNPs_code_only), axis=2)

# Add start of the sentence base vectors
start_codeonly = np.ones(10)
start_typeword = np.ones(13) 
# Broadcast to code only and code + type of word arrays...
x_codeonly_start = np.broadcast_to(start_codeonly, (Gram_shape[0],1,10))
x_typeword_start = np.broadcast_to(start_typeword, (Gram_shape[0], 1, 13))

# Concatenate to create the data arrays to be saved...
Gram_codeonly_data = np.concatenate((x_codeonly_start, Gram_code_only), axis=1)
Gram_typeword_data = np.concatenate((x_typeword_start, Gram_code_typeword), axis=1)

Jabb_codeonly_data = np.concatenate((x_codeonly_start, Jabb_code_only), axis=1)
Jabb_typeword_data = np.concatenate((x_typeword_start, Jabb_code_typeword), axis=1)

OnlyNPs12_codeonly_data = np.concatenate((x_codeonly_start, OnlyNPs12_code_only), axis=1)
OnlyNPs12_typeword_data = np.concatenate((x_typeword_start, OnlyNPs12_code_typeword), axis=1)

OnlyNPs13_codeonly_data = np.concatenate((x_codeonly_start, OnlyNPs13_code_only), axis=1)
OnlyNPs13_typeword_data = np.concatenate((x_typeword_start, OnlyNPs13_code_typeword), axis=1)

OnlyNPs_codeonly_data = np.concatenate((x_codeonly_start, OnlyNPs_code_only), axis=1)
OnlyNPs_typeword_data = np.concatenate((x_typeword_start, OnlyNPs_code_typeword), axis=1)

Wordlist_codeonly_data = np.concatenate((x_codeonly_start, Wordlist_code_only), axis=1)

################## Create criteria database ##################

# Read a list of words from a textfie and store them in a plane list
def list_from_file(filename):
    emptylist = []
    with open(filename) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            # if you need to use the file content as numbers:
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            emptylist += inner_list # to merge lists just sum them up!
    f.close()
    return emptylist
Gram_list = list_from_file('Grammatical_Clean.txt')
Jabb_list = list_from_file('Jabberwocky_Clean.txt')
OnlyNPs12_list = list_from_file('OnlyNPs_1_2_Clean.txt')
OnlyNPs13_list = list_from_file('OnlyNPs_1_3_Clean.txt')
OnlyNPs_list = list_from_file('OnlyNPs_Clean.txt')
WordList_list = list_from_file('WordList_Clean.txt')

Total_list = Gram_list + Jabb_list + OnlyNPs12_list + OnlyNPs13_list + OnlyNPs_list + WordList_list

# Create an array of unique words from Gram_array
def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
unique_words = np.array(getUniqueWords(Total_list))

# Use the amount of different words to determine the dimension on the one-hot vector add 1 cero for the end_of_sentence code
vector_lenght = len(unique_words) + 1
print vector_lenght

# Use the np.eye function to create an array of one-hot vectors to 
# Iterate over and shuffle it
onehot_matrix = np.eye(vector_lenght)
onehot_words_matrix = onehot_matrix[0:-1,:]
np.random.shuffle(onehot_words_matrix)

# Create a dictionary from the list of unique words and one-hot arrays
word_dic = dict(zip(unique_words, onehot_words_matrix))

# From this dictionary asign one-hot vectors to the words in the database
def onehot_database(words_array):
    data_shape = np.array(words_array).shape
    (dim1, dim2) = data_shape[0], data_shape[1]
    onehot_data = np.zeros((data_shape[0], data_shape[1], vector_lenght))
    for i in range(dim1):
        for j in range(dim2):
            onehot_data[i,j] = word_dic[words_array[i,j]]
    return onehot_data

Gram_onehot = onehot_database(Gram_array)
Jabb_onehot = onehot_database(Jabb_array)
OnlyNPs12_onehot = onehot_database(OnlyNPs12_array)
OnlyNPs13_onehot = onehot_database(OnlyNPs13_array)
OnlyNPs_onehot = onehot_database(OnlyNPs_array)
Wordlist_onehot = onehot_database(Wordlist_array)
# Criteria database. Add a end of the sentence colum
end_honehot = onehot_matrix[-1,:]
end_honehot_array = np.broadcast_to(end_honehot, (60,1, len(end_honehot)))

Gram_Y = np.append(Gram_onehot, end_honehot_array, axis = 1)
Jabb_Y = np.append(Jabb_onehot, end_honehot_array, axis = 1)
OnlyNPs12_Y = np.append(OnlyNPs12_onehot, end_honehot_array, axis = 1)
OnlyNPs13_Y = np.append(OnlyNPs13_onehot, end_honehot_array, axis = 1)
OnlyNPs_Y = np.append(OnlyNPs_onehot, end_honehot_array, axis = 1)
Wordlist_Y = np.append(Wordlist_onehot, end_honehot_array, axis = 1)

##################################################################

# Make XY tuples to save in file 
Gram_codeonly_XY = (Gram_codeonly_data, Gram_Y)
Gram_typeword_XY = (Gram_typeword_data, Gram_Y)

Jabb_codeonly_XY = (Jabb_codeonly_data, Jabb_Y)
Jabb_typeword_XY = (Jabb_typeword_data, Jabb_Y)

OnlyNPs12_codeonly_XY = (OnlyNPs12_codeonly_data, OnlyNPs12_Y)
OnlyNPs12_typeword_XY = (OnlyNPs12_typeword_data, OnlyNPs12_Y)

OnlyNPs13_codeonly_XY = (OnlyNPs13_codeonly_data, OnlyNPs13_Y)
OnlyNPs13_typeword_XY = (OnlyNPs13_typeword_data, OnlyNPs13_Y)

OnlyNPs_codeonly_XY = (OnlyNPs_codeonly_data, OnlyNPs_Y)
OnlyNPs_typeword_XY = (OnlyNPs_typeword_data, OnlyNPs_Y)

Wordlist_codeonly_XY = (Wordlist_codeonly_data, Wordlist_Y)

##save the completely random code array and the random code + type 
## of word arrays in a pickle file for later RNN training...
## 'wb' is for write in binary
"""
cPickle.dump(Gram_codeonly_XY, open('Gram_codeonly.cPickle', 'wb'))
#cPickle.dump(Gram_typeword_XY, open('Gram_typeword.cPickle', 'wb'))

cPickle.dump(Jabb_codeonly_XY, open('Jabb_codeonly.cPickle', 'wb'))
#cPickle.dump(Jabb_typeword_XY, open('Jabb_typeword.cPickle', 'wb'))

cPickle.dump(OnlyNPs12_codeonly_XY, open('OnlyNPs12_codeonly.cPickle', 'wb'))
#cPickle.dump(OnlyNPs12_typeword_XY, open('OnlyNPs12_typeword.cPickle', 'wb'))

cPickle.dump(OnlyNPs13_codeonly_XY, open('OnlyNPs13_codeonly.cPickle', 'wb'))
#cPickle.dump(OnlyNPs13_typeword_XY, open('OnlyNPs13_typeword.cPickle', 'wb'))

cPickle.dump(OnlyNPs_codeonly_XY, open('OnlyNPs_codeonly.cPickle', 'wb'))
#cPickle.dump(OnlyNPs_typeword_XY, open('OnlyNPs_typeword.cPickle', 'wb'))
"""
cPickle.dump(Wordlist_codeonly_XY, open('Wordlist_codeonly.cPickle', 'wb'))

##Load training data ('rb' is for read from binary)
#import cPickle
x, y = cPickle.load(open('Wordlist_codeonly.cPickle', 'rb'))
print x.shape, y.shape
