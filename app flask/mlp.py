from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def mlp(p1, p2, classificador, randoms):
    iris = load_iris()
    X = iris.data # caracteristica
    y = iris.target # rotulos

    # Divisão dos Dados em Treinamento e Teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int(randoms))

    # Escolha do Algoritmo de Machine Learning a partir da variavel "classificador".
    clf = MLPClassifier(hidden_layer_sizes=(int(p1),), activation=(p2))
    clf.fit(X_train, y_train)

    # Teste / Previsão do Modelo.
    y_pred = clf.predict(X_test)

    # chama as funções de cada métrica
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
        
    # Análise dos Resultados.
    cm = confusion_matrix(y_test, y_pred)

    classes = iris.target_names.tolist()

    # cria a matriz de confusão para display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # plota a imagem para ser salva
    disp.plot()
    # salva a imagem para ser usada no html
    plt.savefig("static/img")
    return accuracy, precision, recall, f1