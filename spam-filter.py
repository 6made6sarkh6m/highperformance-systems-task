from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB


x = df.text
y = df.label_num



X_train, X_test, y_train, y_test = train_test_split(X,y)

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train.values)

classifier = MultinomialNB()
targets = y_train.values

classifier.fit(counts, targets)


# здесь вам нужно создать свой датасет для проверки. Это должен быть обычный массив examples из двух индексов, где в первом будет спам, а во втором сообщение

example_count = vectorizer.transform(examples)
predictions = classifier.predict(example_count)
