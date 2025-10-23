import json

def make_vocab_file(data, vocab_column="ipa_transcription", output_file="vocab.json"):
    print(f"making vocab file: {output_file}")
    def extract_all_chars(batch):
        all_text = " ".join([" ".join(x) for x in batch[vocab_column]])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(output_file, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

