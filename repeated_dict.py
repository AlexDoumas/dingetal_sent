#repeated_dict
'''reads a dataset in a txt file and return a list with repeated words and frecuency'''

def repeated_dict(string_or_list):
    d = dict()
    d_repeated = dict()
    for c in string_or_list:
        if c not in d:
            d[c] = 1
        else:
            d[c] += 1
    for c in d:
        if d[c] > 1:
            d_repeated[c] = d[c]
    return d_repeated

# Read a list of words from a textfie and store them in a list
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

Gram_repeated = repeated_dict(Gram_list)
print len(Gram_repeated)
print Gram_repeated