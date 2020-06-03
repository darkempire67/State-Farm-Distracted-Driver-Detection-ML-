# State-Farm-Distracted-Driver-Detection-ML-
For this project I used Python and Keras on top of TensorFlow and coded a simple detection Convolutional Neural Network to predict the driver's actions in the image input. (If the photo shows a driver texting with the left hand) my model "should" be able to predict that label. There are 10 labels all depicting different actions by drivers. The data set was 4gb, and is given by kaggle.https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview/description  The data set for training does not overlap with the testing set. The validation accuracy shown is not accurate though, after careful analysis the model predicted about 60 to maybe top 70 percent of the images with the correct label.

Kaggle description -State Farm hopes to improve these alarming statistics, and better insure their customers, by testing whether dashboard cameras can automatically detect drivers engaging in distracted behaviors. Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?

Layer (type)                 Output Shape              Param #   

=================================================================
conv2d_9 (Conv2D)            (None, 120, 160, 32)      320       
_________________________________________________________________
activation_15 (Activation)   (None, 120, 160, 32)      0         
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 40, 53, 32)        0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 40, 53, 32)        9248      
_________________________________________________________________
activation_16 (Activation)   (None, 40, 53, 32)        0         
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 19, 26, 32)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 19, 26, 32)        9248      
_________________________________________________________________
activation_17 (Activation)   (None, 19, 26, 32)        0         
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 9, 12, 32)         0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 9, 12, 32)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 3456)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 528)               1825296   
_________________________________________________________________
activation_18 (Activation)   (None, 528)               0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 528)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 10)                5290      
_________________________________________________________________
activation_19 (Activation)   (None, 10)                0         

=================================================================
Total params: 1,849,402
Trainable params: 1,849,402
Non-trainable params: 0

# I did not use the full data set.
Train on 13509 samples, validate on 1501 samples
Epoch 1/3
13509/13509 [==============================] - 1727s 128ms/sample - loss: 1.1920 - acc: 0.5860 - val_loss: 0.2377 - val_acc: 0.9374
Epoch 2/3
13509/13509 [==============================] - 1732s 128ms/sample - loss: 0.2919 - acc: 0.9081 - val_loss: 0.1126 - val_acc: 0.9660
Epoch 3/3
13509/13509 [==============================] - 1581s 117ms/sample - loss: 0.1659 - acc: 0.9473 - val_loss: 0.0769 - val_acc: 0.9800

