import spacy
from spacy.tokens import Doc

nlp = spacy.load('zh_core_web_sm')



class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


text = "我 来 自 北 京 市 的 北 京 交 通 大 o k"
text_list = text.split()
doc = nlp(text)

# print(doc)
for token in doc:
    print(token, text_list[token.head.i], text_list[token.i])
    print(token.head.i, token.i)
