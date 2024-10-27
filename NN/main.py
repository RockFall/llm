import numpy as np
from neural_network import NN

# Exemplo de uso
x_train = np.array([[1,1,1],[0,0,0],[1,0,1],[0,1,0],[1,1,0],[0,0,1],[1,0,0],[0,1,1]])
y_train = np.array([1,0,0,0,0,0,0,0])

x_test = np.array([[0,0,0],[1,0,1],[0,1,0],[1,1,0],[1,1,1],[0,0,1],[1,0,0],[0,1,1]])
y_test = np.array([0,0,0,0,1,0,0,0])

nn = NN()
nn.fit(x_train, y_train)

y_pred = nn.predict(x_test)
print(y_pred)
