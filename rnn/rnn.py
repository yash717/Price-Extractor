from keras.models import load_model
import numpy as np
import pickle
from keras_preprocessing.sequence import pad_sequences

# Load the model
loaded_model = load_model('price_coupon_model.h5')
# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


while True:
    # Define the input string
    text = input('Enter the string: ')
    if len(text) > 0 :
        if text == 'q':
            break
        string = text.replace(',', '').lower()
        # Tokenize the input string
        sequence = tokenizer.texts_to_sequences([string.lower()])

        # Pad the sequence
        max_length = 35
        padded_sequence = pad_sequences(sequence, maxlen=max_length)


        # Predict the label
        prediction = loaded_model.predict(padded_sequence)

        # Get the label with the highest probability
        label = np.argmax(prediction)

        # Print the label
        if label == 0:
            print('No Coupon or Price Found.')
        if label == 1:
            print('Only Price Found.')
        if label == 2:
            print('Only Coupon Found.')
        if label == 3:
            print('Both Coupon and Price Found.')
        print(label)
    else:
        print('no input given')
