from easse.bleu import corpus_bleu
from easse.sari import corpus_sari, get_corpus_sari_operation_scores

original_file = "refs/newsela-en-valid.simple"
sys_out_file = "multi/prediction-multi-all-en-sim.txt"
ref1_file = "refs/newsela-en-valid.simple"

original = open(original_file).readlines()
syst = open(sys_out_file).readlines()
ref1 = open(ref1_file).readlines()

sari_val = corpus_sari(orig_sents=original, sys_sents=syst, refs_sents=[ref1])
print("File path: ", sys_out_file)
print("SARI:", round(sari_val, 2))
add, keep, dele = get_corpus_sari_operation_scores(orig_sents=original,
                                                   sys_sents=syst,
                                                   refs_sents=[ref1])
print("add, del, kept: ", round(add, 2), round(dele, 2), round(keep, 2))
bleu_val = corpus_bleu(sys_sents=syst, refs_sents=[ref1])
print("BLEU: ", round(bleu_val, 2))

