import time
from smolbpe import BasicTokenizer

text = open("./taylorswift.txt", "r", encoding="utf-8").read()
print(len(text))

start = time.time()

tokenizer = BasicTokenizer()
tokenizer.train(text, 512, True)

end = time.time()

print(f"training took {end - start:.2f} seconds")
