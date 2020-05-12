# Read the positive and negative words into sets
pos_words = set()
neg_words = set()

with open('positive-words.txt', 'r') as file_pos:
    for line in file_pos:
        if line.startswith(';'):
            continue
        elif line.startswith(' '):
            continue
        else:
            word = line.strip("\n")
            pos_words.add(word)

with open('negative-words.txt', 'r', encoding="ISO-8859-1") as file:
    for line in file:
        if line.startswith(';'):
            continue
        elif line.startswith(' '):
            continue
        else:
            word = line.strip("\n")
            neg_words.add(word)

pos_words.remove('')
neg_words.remove('')
