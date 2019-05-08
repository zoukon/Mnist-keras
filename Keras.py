import sys
import keras
import pandas as pd
import numpy as np
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten


def run_model(epochs = 25, batch_size = 128):
    #load mnist dataset
    #(X_train, y_train), (X_test, y_test) = mnist.load_data() 
    
    a = np.load("mnist.npz")
    X_test = a['x_test']
    y_test = a['y_test']

    X_train = a['x_train']
    y_train = a['y_train']




    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
    
    
    
    #more reshaping
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)
    print('X_test shape:', X_test.shape) #X_train shape: (10000, 28, 28, 1)
    
    #set number of categories
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)
    
    
    
    
    
    ##model building
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    
    #32 convolution filters used each of size 3x3
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #64 convolution filters used each of size 3x3
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    #fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    #one more dropout for convergence' sake  
    model.add(Dropout(0.5))
    #output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_category, activation='softmax'))
    
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    
    #We use adam as our optimizer
    #categorical ce since we have multiple classes (10) 
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])
    
    
    #model training
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, y_test))
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1]) 
    
    
    Y_predicted = model.predict(X_test)
    pred_label = Y_predicted.argmax(axis = 1)
    image_id = range(1,len(Y_predicted)+1)
    df = {'ImageId':image_id,'Label':pred_label}
    df = pd.DataFrame(df)
    df.to_csv('results.csv',index = False)
    
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

if __name__ == '__main__':
    if(len(sys.argv) == 3):
        run_model(int(sys.argv[1]), int(sys.argv[2]) )
    elif(len(sys.argv) == 2):
        run_model(int(sys.argv[1])) 
    else:
        run_model()


#https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
        