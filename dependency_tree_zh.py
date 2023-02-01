import spacy
nlp = spacy.load('zh_core_web_sm')

text = "我 来 自 北 京 市 的 北 京 交 通 大 o k"
text_list = text.split()
doc = nlp(text)

# print(doc)
for token in doc:
    print(token, text_list[token.head.i], text_list[token.i])
    print(token.head.i, token.i)
