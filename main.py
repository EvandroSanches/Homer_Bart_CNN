import keras.optimizers.schedules
from keras.utils import image_dataset_from_directory
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, RandomZoom, RandomFlip, RandomRotation, Rescaling
from keras.models import Sequential, load_model
from scikeras.wrappers import KerasClassifier
from keras.preprocessing import image
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt


epochs = 30
batch_size = 10

#Carrega e trata dados do data set
#Bart - 0
#Homer - 1
def CarregaDados():
    dados = image_dataset_from_directory(('training_set'), image_size=(128,128), batch_size=269)

    previsores = np.empty([0,128,128,3])
    classe = np.empty([0,1])

    for imagem, label in dados:
        previsores = np.append(previsores, values=imagem, axis=0)
        label = np.expand_dims(label, axis=1)
        classe = np.append(classe, values=label, axis=0)

    return previsores, classe


#Criando modelo de rede neural
def CriaRede():
    modelo = Sequential([
        Rescaling(scale= 1./255, input_shape=(128,128,3)),
        RandomFlip('horizontal'),
        RandomZoom(0.3),
        RandomRotation(0.2)
    ])

    modelo.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    modelo.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    modelo.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    modelo.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    modelo.add(Flatten())

    modelo.add(Dense(units=450, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=450, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=1, activation='sigmoid'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0009,
        decay_steps=750,
        decay_rate=0.0005
    )

    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler), loss='binary_crossentropy', metrics=['accuracy'])

    return modelo

def Treinamento():
    previsores, classe = CarregaDados()

    modelo = KerasClassifier(build_fn=CriaRede, epochs=epochs, batch_size=batch_size)

    result = cross_val_score(estimator=modelo, X=previsores, y=classe, cv=10)

    print('Resultado:'+str(result))
    print('Média:'+str(result.mean()))
    print('Desvio Padrão:'+str(result.std()))

def GeraModelo():
    previsores, classe = CarregaDados()

    modelo = CriaRede()

    result = modelo.fit(previsores, classe, batch_size=batch_size, epochs=epochs)

    modelo.save('Modelo.0.1')

    plt.plot(result.history['loss'])
    plt.title('Relação de Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()

    plt.plot(result.history['accuracy'])
    plt.title('Relação de Acuracia')
    plt.xlabel('Épocas')
    plt.ylabel('Acuracia')
    plt.show()

def Previsao(caminho):
    previsor = image.load_img(caminho, target_size=(128, 128))

    modelo = load_model('Modelo.0.1')

    previsor = image.img_to_array(previsor)
    previsor = np.expand_dims(previsor, axis=0)

    resultado = modelo.predict(previsor)

    if resultado > 0.5:
        return 'Homer'
    else:
        return 'Bart'

result = Previsao('training_set/homer/homer24.bmp')
print(result)
