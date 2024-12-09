from model import *


def generate_response(question: str) -> str | None:
    X = bag_of_words(tokenize(question), all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float()

    output = model(X)
    _, predicted = torch.max(output, 1)
    tag = tags[predicted.item()]

    for datum in data:
        if tag == datum['tag']:
            return random.choice(datum['responses'])

    return None


model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

if __name__ == '__main__':
    while True:
        question = input('Question: ')
        print(generate_response(question))
