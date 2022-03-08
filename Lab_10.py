# [1]
import zipfile

zipFile = zipfile.ZipFile("review_polarity.zip")

pos_files = [f for f in zipFile.namelist() if '/pos/cv' in f]
neg_files = [f for f in zipFile.namelist() if '/neg/cv' in f]

pos_files.sort()
neg_files.sort()

print("Recenzii pozitive: " + str(len(pos_files)) + "; Recenzii negative: " + str(len(neg_files)))

# Raspunsul asteptat: "Recenzii pozitive: 1000; Recenzii negative: 1000"
assert(len(pos_files) == 1000 and len(neg_files) == 1000)

# [2]

tr_pos_no = int(.8 * len(pos_files))
tr_neg_no = int(.8 * len(neg_files))

from random import shuffle
shuffle(pos_files)
shuffle(neg_files)

pos_train = pos_files[:tr_pos_no] # Recenzii pozitive pentru antrenare
pos_test  = pos_files[tr_pos_no:] # Recenzii pozitive pentru testare
neg_train = neg_files[:tr_neg_no] # Recenzii negative pentru antrenare
neg_test  = neg_files[tr_neg_no:] # Recenzii negative pentru testare

# [3]

STOP_WORDS = []
#STOP_WORDS = [line.strip() for line in open("Lab12-stop_words")]

import re

POS = 0
NEG = 1

def parse_document(path):
    for word in re.findall(r"[-\w']+", zipFile.read(path).decode("utf-8")):
        if len(word) > 1 and word not in STOP_WORDS:
            yield word

def count_words():
    vocabulary = {}
    pos_words_no = 0
    neg_words_no = 0
    
    # ------------------------------------------------------
    # <TODO 1> numrati aparitiile in documente pozitive si
    # in documente negative ale fiecarui cuvant, precum si numarul total
    # de cuvinte din fiecare tip de recenzie
    for pos_file in pos_train:
        for word in parse_document(pos_file):
            pos_words_no += 1
            if word in vocabulary:
                vocabulary[word][POS] += 1
            else:
                vocabulary[word] = [1, 0]

    for neg_file in neg_train:
        for word in parse_document(neg_file):
            neg_words_no += 1
            if word in vocabulary:
                vocabulary[word][NEG] += 1
            else:
                vocabulary[word] = [0, 1]
    # ------------------------------------------------------

    return (vocabulary, pos_words_no, neg_words_no)

# -- VERIFICARE --
training_result_words = count_words()

(voc, p_no, n_no) = training_result_words
print("Vocabularul are ", len(voc), " cuvinte.")
print(p_no, " cuvinte in recenziile pozitive si ", n_no, " cuvinte in recenziile negative")
print("Cuvantul 'beautiful' are ", voc.get("beautiful", (0, 0)), " aparitii.")
print("Cuvantul 'awful' are ", voc.get("awful", (0, 0)), " aparitii.")

# Daca se comentează liniile care reordonează aleator listele cu exemplele pozitive și negative,
# rezultatul așteptat este:
#
# Vocabularul are  44895  cuvinte.
# 526267  cuvinte in recenziile pozitive si  469812  cuvinte in recenziile negative
# Cuvantul 'beautiful' are  (165, 75)  aparitii.
# Cuvantul 'awful' are  (16, 89)  aparitii.

# [4]

from math import log

def predict(params, path, alpha = 1):
    (vocabulary, pos_words_no, neg_words_no) = params
    laplace = len(vocabulary.keys()) * alpha
    log_pos = log(0.5)
    log_neg = log(0.5)
    
    # ----------------------------------------------------------------------
    # <TODO 2> Calculul logaritmilor probabilităților
    for word in parse_document(path):
        cnt = vocabulary.get(word, [0, 0])

        log_pos += log((cnt[POS] + alpha) / (pos_words_no + laplace))
        log_neg += log((cnt[NEG] + alpha) / (neg_words_no + laplace))
    # ----------------------------------------------------------------------

    if log_pos > log_neg:
        return "pos", log_pos
    else:
        return "neg", log_neg

# -- VERIFICARE --
print(zipFile.read(pos_test[14]).decode("utf-8"))
predict(training_result_words, pos_test[14])

# Daca se comentează liniile care reordonează aleator listele cu exemplele pozitive și negative,
# rezultatul așteptat este:
#
# ('pos', -1790.27088356391) pentru un film cu Hugh Grant și Julia Roberts (o mizerie siropoasă)
#
# Recenzia este clasificată corect ca fiind pozitivă.