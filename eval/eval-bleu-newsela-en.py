from easse.bleu import corpus_bleu

original_file = "refs/newsela-en-valid.simple"
sys_out = "multi/prediction-adapter-fusion-tr-para-simp.txt"

original = open(original_file).readlines()[:1000]
syst = open(sys_out).readlines()

print("System file path: ", sys_out)
print("Against: ", original_file)
bleu_val = corpus_bleu(sys_sents=syst, refs_sents=[original])
print("BLEU: ", round(bleu_val, 2))

