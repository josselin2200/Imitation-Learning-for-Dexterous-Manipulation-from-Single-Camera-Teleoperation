import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


data_dict = pickle.load(open('./data.pickle', 'rb'))

#print(data_dict.keys())
#print(data_dict.values())

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

"""

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
"""




models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=500),
    "Logistic Regression (Multinomial)": LogisticRegression(solver='lbfgs', max_iter=500)
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f"{name}: {score * 100:.2f}% accuracy")

# Sauvegarde du meilleur modèle (par exemple, le plus performant)
best_model = max(models.items(), key=lambda x: accuracy_score(x[1].predict(x_test), y_test))[1]

with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)


