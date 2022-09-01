from collections import defaultdict

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Paraphraser:

    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("models/CORD19_t5_finetuned_gpu8")
        self.model = T5ForConditionalGeneration.from_pretrained("models/CORD19_t5_finetuned_gpu8").cpu() #ceshine/t5-paraphrase-paws-msrp-opinosis

    def get_response(self, input_text, num_return_sequences):
        generated = self.model.generate(self.tokenizer.encode("paraphrase: " + input_text, return_tensors="pt").cpu(),
            num_beams=25, num_return_sequences=num_return_sequences, max_length=64)
        tgt_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return tgt_text

class PegasusParaphraser:
    def __init__(self):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.torch_device)

    def get_response(self, input_text, num_return_sequences):
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch, max_length=60, num_beams=10, num_return_sequences=num_return_sequences,temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text


sentencemodel = SentenceTransformer('bert-large-nli-mean-tokens')

with open("affirmatives-new.txt") as file:
    modelo = T5Paraphraser()
    counter = 0
    print("topic-number;topic;paraphrase1;paraphrase2;...")
    for line in file.readlines():#[:10]:
        line = line[0].upper() + line.strip()[1:] + "."
        #print("\nSentence: " + line)
        counter+=1
        print(str(100+counter)+";"+line, end='')
        sentence1_embedding = sentencemodel.encode(line, show_progress_bar=False)
        paraphrases = defaultdict(list)
        for paraphrase in modelo.get_response(line, 8):
            paraphrase=paraphrase.strip("\"' ")
            sentence2_embedding = sentencemodel.encode(paraphrase, show_progress_bar=False)
            simil = 1 - cosine(sentence1_embedding, sentence2_embedding)
            paraphrases[round(simil, 3)].append(paraphrase)
        for score, phraselist in sorted(paraphrases.items(), reverse=True):
            if 0.80 < score < 1.0:
                phraseset=set()
                for phrase in phraselist:
                    if phrase.lower() not in phraseset:
                        #print("\t" + phrase + "\t" + str(score))
                        print(";"+phrase,end='')
                        phraseset.add(phrase.lower())
        print()
