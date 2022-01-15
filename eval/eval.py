from easse.bleu import corpus_bleu
from easse.cli import report
from easse.sari import corpus_sari, get_corpus_sari_operation_scores

original_file = "complex.txt"
sys_out = "prediction-model0.txt"
ref1_file = "simp1.txt"
ref2_file = "simp2.txt"
original = open(original_file).readlines()
syst = open(sys_out).readlines()
ref1 = open(ref1_file).readlines()
ref2 = open(ref2_file).readlines()

sari_val = corpus_sari(orig_sents=original, sys_sents=syst, refs_sents=[ref1, ref2])
print("SARI:", round(sari_val, 2))
add, keep, dele = get_corpus_sari_operation_scores(orig_sents=original,
                                                   sys_sents=syst,
                                                   refs_sents=[ref1, ref2])
print("add, del, kept: ", round(add, 2), round(dele, 2), round(keep, 2))
bleu_val = corpus_bleu(sys_sents=syst, refs_sents=[ref1, ref2])
print("BLEU: ", round(bleu_val,2))
# report(
#     "custom",%3CmxGraphModel%3E%3Croot%3E%3CmxCell%20id%3D%220%22%2F%3E%3CmxCell%20id%3D%221%22%20parent%3D%220%22%2F%3E%3CmxCell%20id%3D%222%22%20value%3D%22%26lt%3Bb%26gt%3B%26lt%3Bi%26gt%3BAuxiliary%20Task%20n%26lt%3B%2Fi%26gt%3B%26lt%3B%2Fb%26gt%3B%22%20style%3D%22rounded%3D0%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3BfontSize%3D14%3BstrokeWidth%3D2%3BfillColor%3D%23d5e8d4%3BstrokeColor%3D%2382b366%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22540%22%20y%3D%22640%22%20width%3D%22130%22%20height%3D%2260%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3C%2Froot%3E%3C%2FmxGraphModel%3E
#     sys_sents_path=sys_out,
#     orig_sents_path=original_file,
#     refs_sents_paths="simp1.txt,simp2.txt",
#     lowercase=False
# )

# easse report --orig_sents_path complex.txt --sys_sents_path prediction-model1.txt --test_set "custom" \
# --refs_sents_paths "simp1.txt,simp2.txt"  --report_path ./newexp-model25P.html
