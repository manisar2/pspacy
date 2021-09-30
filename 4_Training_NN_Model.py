# Why Updating the Model
    # Essential for classification
    # Very useful for NER
    # Less critical for POS tagging and dependency parsing

################################################################################
# 1.1 Creating Training Data
################################################################################
import json
import en_core_web_sm
from spacy.matcher import Matcher
from spacy.lang.en import English
with open("iphone.json", encoding="utf8") as f: TEXTS = json.loads(f.read())

nlp = English()
matcher = Matcher(nlp.vocab)
pattern1 = [{"LOWER": "iphone"}, {"LOWER": "x"}]
pattern2 = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]
matcher.add("GADGET", [pattern1, pattern2])
for doc in nlp.pipe(TEXTS): print([doc[start:end] for match_id, start, end in matcher(doc)])
################################################################################

# 1.2
    # Let’s use the match patterns we’ve created in the previous exercise to 
    # bootstrap a set of training examples. A list of sentences is available as 
    # the variable TEXTS.
    # - Create a doc object for each text using nlp.pipe.
    # - Match on the doc and create a list of matched spans.
    # - Get (start character, end character, label) tuples of matched spans.
    # - Format each example as a tuple of the text and a dict, mapping 
    #   "entities" to the entity tuples.
    # - Append the example to TRAINING_DATA and inspect the printed data.
################################################################################
TRAINING_DATA = []
for doc in list(nlp.pipe(TEXTS)):
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    entities = [(span.start_char, span.end_char, "GADGET") for span in spans]
    training_example = (doc.text, {"entities": entities})
    TRAINING_DATA.append(training_example)
print(*TRAINING_DATA, sep="\n")
################################################################################

################################################################################
# 2.1 Set up the pipeline
################################################################################
import spacy
nlp = spacy.blank("en") # new nlp with just the tokenizer
# ner = nlp.create_pipe("ner")
nlp.add_pipe("ner").add_label("GADGET")

# ner.add_label("GADGET")
################################################################################

################################################################################
# 2.2 Building a Training Loop
    # - Call nlp.begin_training, create a training loop for 10 iterations and 
    #   shuffle the training data.
    # - Create batches of training data using spacy.util.minibatch and iterate 
    #   over the batches.
    # - Convert the (text, annotations) tuples in the batch to lists of texts and 
    #   annotations.
    # - For each batch, use nlp.update to update the model with the texts and 
    #   annotations.
################################################################################
import random
from spacy.training import Example
nlp.begin_training()
for itn in range(10): # epochs
    random.shuffle(TRAINING_DATA)
    losses = {}

    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        # texts = [text for text, entities in batch]
        # annotations = [entities for text, entities in batch]
        # nlp.update(texts, annotations, losses=losses)
        texts, annotations = zip(*batch)
        docs = [nlp.make_doc(text) for text in texts]
        examples = [Example.from_dict(docs[i], annotations[i]) for i in range(len(docs))]
        # print(examples)
        nlp.update(examples, losses=losses)
        
        print(losses)
################################################################################

# 3. Good Data vs. Bad Data
################################################################################
TRAINING_DATA = [
    (
        "i went to amsterdem last year and the canals were beautiful",
        {"entities": [(10, 19, "GPE")]},
    ),
    (
        "You should visit Paris once in your life, but the Eiffel Tower is kinda boring",
        {"entities": [(17, 22, "GPE")]},
    ),
    ("There's also a Paris in Arkansas, lol", {"entities": [(15, 20, "GPE"), (24, 32, "GPE")]}),
    (
        "Berlin is perfect for summer holiday: lots of parks, great nightlife, cheap beer!",
        {"entities": [(0, 6, "GPE")]},
    ),
]
################################################################################

# 4. Training Multiple Labels
# 4.1
################################################################################
TRAINING_DATA = [
    (
        "Reddit partners with Patreon to help creators build communities",
        {"entities": [(0, 5, "WEBSITE"), (21, 28, "WEBSITE")]},
    ),
    ("PewDiePie smashes YouTube record", {"entities": [(18, 25, "WEBSITE")]}),
    (
        "Reddit founder Alexis Ohanian gave away two Metallica tickets to fans",
        {"entities": [(0, 6, "WEBSITE")]},
    ),
    # And so on...
]
################################################################################

# 4.2
################################################################################
TRAINING_DATA = [
    (
        "Reddit partners with Patreon to help creators build communities",
        {"entities": [(0, 6, "WEBSITE"), (21, 28, "WEBSITE")]},
    ),
    (   "PewDiePie smashes YouTube record", 
        {"entities": [(0, 9, "PERSON"), (18, 25, "WEBSITE")]},),
    (
        "Reddit founder Alexis Ohanian gave away two Metallica tickets to fans",
        {"entities": [(0, 6, "WEBSITE"), (15, 29, "PERSON")]},
    ),
    # And so on...
]
################################################################################

# Learnings
# * Extract linguistic features: pos, dep_, ner
# * Work with pre-trained statistical models, such as en_core_web_sm
# * Writing match rules using Matcher and PhraseMatcher
# * Working with data structures
# * semantic similarities using word vectors
# * custom pipeline components and extensions
# * Performance tuning
# * Create training data
# * Training
