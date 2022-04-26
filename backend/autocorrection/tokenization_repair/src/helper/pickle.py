import pickle


def dump_object(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_object(path):
    # path = "F:\\LVTN\\SpellingCorrectionApp\\backend\\tokenization_repair\\data\\estimators\\lm\\unilm\\specification.pkl"
    with open(path, "rb") as file:
        print(path)
        tmp = pickle.load(file)
        return tmp
        # return pickle.load(file)
