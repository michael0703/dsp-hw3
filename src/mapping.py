import sys
from collections import defaultdict
f_in = open(sys.argv[1], "r", encoding = "big5-hkscs")
f_out = open(sys.argv[2], "w", encoding = "big5-hkscs")

count = 0
inverted_map = defaultdict(list)
for line in f_in.readlines():
	line = line[:-1]
	line = line.split()
	term = line[0]
	spell = line[1]
	
	multi_spell = spell.split("/")
	multi_head = [x[0] for x in multi_spell]	
	inverted_map[term].append(term)
	for head in multi_head:
		if term not in inverted_map[head]:
			inverted_map[head].append(term)
	print(f"X:{term} / Y:{multi_spell}")
	count += 1
	#if count > 10:
	#	break

for key, terms in inverted_map.items():
	f_out.write(f"{key} {' '.join(terms)}\n")

