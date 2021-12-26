from easse.bleu import corpus_bleu
from easse.cli import report
from easse.sari import corpus_sari, get_corpus_sari_operation_scores

original_file = "complex.txt"
sys_out = "prediction-model1.2.txt"
ref1_file = "simp1.txt"
ref2_file = "simp2.txt"
original = open(original_file).readlines()
syst = open(sys_out).readlines()
ref1 = open(ref1_file).readlines()
ref2 = open(ref2_file).readlines()

print("SARI Score using ref: ", corpus_sari(orig_sents=original,
                                            sys_sents=syst,
                                            refs_sents=[ref1, ref2]))

print("SARI components: add, keep ,del ", get_corpus_sari_operation_scores(orig_sents=original,
                                                                           sys_sents=syst,
                                                                           refs_sents=[ref1, ref2]))
print("BLEU: ", corpus_bleu(sys_sents=syst, refs_sents=[original]))

# report(
#     "custom",
#     sys_sents_path=sys_out,
#     orig_sents_path=original_file,
#     refs_sents_paths=ref1_file,
#     lowercase=False
# )