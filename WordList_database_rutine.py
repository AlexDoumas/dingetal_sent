import numpy as np
import cPickle

##Read a list of words from a textfie and store them in an array. I'll use 
##the shape of this array to create the lists of codes to feed the RNN
##the files have to use '   ' as delimiter
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
##get array
Gram_array = array_from_file('Grammatical_Clean.txt')
##get shape
Gram_shape = Gram_array.shape
##create codes for type of words in the grammatical list
adj = [0., 0., 1.]
noun = [0., 1., 0.]
verb = [1., 0., 0.]
##create base pattern for database
Gram_patt = np.array([adj, noun, verb, noun])
#get modify shape for broadcasting...
Gram_broadcast = (Gram_shape[0],Gram_shape[1], 3)
##broadcast base pattern to the database shape
Gram_typeword = np.broadcast_to(Gram_patt, Gram_broadcast)

##create array of non repeating random vector with the same shape as the
##database. Also, none of the codes should be equal to 0
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
##add type of word code to every entry of gram_code_only
Gram_code_typeword = np.concatenate((Gram_typeword, Gram_code_only), axis=2)

#crate array of indices to sample from and shuffe it
def shuffled_indexes(raws, cols):
    from random import shuffle
    indlist =[]
    for i in range (raws):
        for j in range(cols):
            x = [i,j]
            indlist.append(x)
    shuffle(indlist)
    return np.array(indlist).reshape(60,4,2)

shuffled_indx = shuffled_indexes(60, 4)
#print shuffled_indx.shape

#function for looping throught the shuffled list and getting an
#array of shape (60, 4, x) with words of codes....
def data_shuffler(data_array, index_array):
    shape = data_array.shape
    a,b = shape[0],shape[1]
    if len(shape)>2:
        c = shape[2]
        shufled_data = np.zeros((a,b,c))
    else:
        #empty array of strings with max 10 letters
        shufled_data = np.empty([a, b], dtype="S10")
    for h in range(a):
        for i in range(b):
            (x,y) = index_array[h,i]
            shufled_data[h,i] = data_array[x,y]
    return shufled_data

WordList_codeonly = data_shuffler(Gram_code_only, shuffled_indx)
WordList_typeword = data_shuffler(Gram_code_typeword, shuffled_indx)
WordList_array = data_shuffler(Gram_array, shuffled_indx)

#ADD THE END-OF-SENENCE CODES BEFORE SAVING
##add end of the sentence base vectors
end_codeonly = np.zeros(10)
end_typeword = np.zeros(13) 
## broadcast to code only and code + type of word arrays...
WordList_codeonly_end = np.broadcast_to(end_codeonly, (Gram_shape[0],1,10))
WordList_typeword_end = np.broadcast_to(end_typeword, (Gram_shape[0], 1, 13))


##concatenate to create the final arrays to be saved...
WordList_codeonly_data = np.concatenate((WordList_codeonly, WordList_codeonly_end), axis=1)

WordList_typeword_data = np.concatenate((WordList_typeword, WordList_typeword_end), axis=1)

##save the completely random code array and the random code + type 
## of word arrays in a pickle file for later RNN training...
## 'wb' is for write in binary
cPickle.dump(WordList_codeonly_data, open('WordList_codeonly.cPickle', 'wb'))
cPickle.dump(WordList_typeword_data, open('WordList_typeword.cPickle', 'wb'))


#save WordList_array as a tab-delimited text file
np.savetxt('WordListPython.txt', WordList_array, fmt='%s', delimiter='\t')
















































