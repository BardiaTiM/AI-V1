import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

df = pd.read_csv('personality.csv', names=['input', 'output'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available

import json

def load_dataset(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    sentence_pairs = []
    for dialogue in data:
        dialogue_turns = dialogue['dialogue']
        for i in range(1, len(dialogue_turns)):
            sentence_pairs.append((dialogue_turns[i-1]['data']['utterance'], dialogue_turns[i]['data']['utterance']))
    return sentence_pairs

# Load the data
sentence_pairs = load_dataset("kvret_train_public.json")



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Vocab:
    def __init__(self):
        self.word2index = {"<SOS>": 0, "<EOS>": 1}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# Dummy sentence pairs: (input, target)
# sentence_pairs = [
#     ("hello how are you", "I am fine thank you"),
#     ("what is your name", "my name is AI"),
#     ("where do you live", "I live in the cloud"),
# ]

# sentence_pairs = list(zip(df['input'].tolist(), df['output'].tolist()))

# Build the vocab from the sentence pairs
vocab = Vocab()
for pair in sentence_pairs:
    vocab.add_sentence(pair[0])
    vocab.add_sentence(pair[1])


import torch.nn.functional as F

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[vocab.word2index['<SOS>']]]).to(device) 
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == vocab.word2index["<EOS>"]:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    print("Training started")  # Add this line
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(sentence_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        if iter % print_every == 0:
            print(f'Iter: {iter}, Loss: {loss:.4f}')  # Uncomment this line
    print("Training finished")  # Add this line


def tensor_from_sentence(sentence):
    indexes = [vocab.word2index[word] for word in sentence.split(' ')]
    indexes.append(vocab.word2index['<EOS>'])
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(pair[0])
    target_tensor = tensor_from_sentence(pair[1])
    return (input_tensor, target_tensor)




def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(sentence).to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[vocab.word2index['<SOS>']]])

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(input_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == vocab.word2index['<EOS>']:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)
    
def main():
    hidden_size = 256
    encoder1 = Encoder(vocab.n_words, hidden_size).to(device)
    decoder1 = Decoder(hidden_size, vocab.n_words).to(device)

    # Check if saved models exist and load them
    import os

    if os.path.isfile("encoder1.pth") and os.path.isfile("decoder1.pth"):
        print("Loading existing models...")
        encoder1.load_state_dict(torch.load("encoder1.pth"))
        decoder1.load_state_dict(torch.load("decoder1.pth"))
    else:
        print("No pre-trained models found.")

    while True:
        print("Select an option:")
        print("1. Train AI")
        print("2. Talk to AI")
        print("3. Quit")
        user_choice = input("Enter your choice: ")
        if user_choice == '1':
            print("Training new models...")
            train_iters(encoder1, decoder1, 75000)
            print("Saving models...")
            torch.save(encoder1.state_dict(), "encoder1.pth")
            torch.save(decoder1.state_dict(), "decoder1.pth")
        elif user_choice == '2':
            while True:
                input_sentence = input("You: ")
                if input_sentence.lower() == 'quit':
                    break
                output_sentence = evaluate(encoder1, decoder1, input_sentence)
                #if the ai cant find a response
                if output_sentence == '':
                    print("AI: Sorry, I don't understand.")
                print(f"AI: {output_sentence}")
        elif user_choice == '3':
            break
        else:
            print("Invalid choice, please try again.")

# main method
if __name__ == '__main__':
    main()
