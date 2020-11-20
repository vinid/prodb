class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def internal_predict_from_tokens(self, string_ids):
        sample_tokens = vectorize_layer([string_ids])

        prediction = self.model.predict(sample_tokens)
        masked_index = np.where(sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-10:][::-1]
        values = mask_prediction[0][top_indices]

        answers = []
        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p

            answers.append(convert_ids_to_tokens(p))

        return answers

    def on_epoch_end(self, epoch, logs=None):

        with open(testing_file) as filino:
            testing_data = list(map(lambda x: x.strip(), filino.readlines()))

        total = 0
        counter = 0
        for a in testing_data:
            splitted = a.split()

            if len(splitted) >= 18:  # skip some examples
                continue

            to_predict = splitted[-1]
            splitted[-1] = "[mask]"
            joined = " ".join(splitted)
            predictions = self.internal_predict_from_tokens(joined)

            if to_predict in predictions[0:5]:
                counter = counter + 1
            total = total + 1

        print("current results", round(counter / total, 3), counter, total)
