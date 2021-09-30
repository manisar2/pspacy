# Using inbuilt pipeline
################################################################################
from spacy.lang.en import English
nlp = English()

doc = nlp("This is a sentence") # sequence
for token in doc: print(token.text)
print(doc[1])
span = doc[1:3]
print(span.text)
################################################################################

# Using Pre-trained pipeline
################################################################################
# Check https://spacy.io/usage/models for more options
import spacy
# nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()

# pos_          : part of speech
    # dep_      : dependency (subject / object etc.)
    # head      : ~parent token
    # ents      : named entities
    # label_    : type of named entity

doc = nlp("She ate the pizza")
for token in doc: print(token.text, token.pos_, token.dep_, token.head.text)

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents: print(ent.text, ent.label_)

for st in ["GPE", "NNP", "dobj"]: print(spacy.explain(st))

for token in doc:
    # Get the token text, part-of-speech tag and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")

for ent in doc.ents: print(ent.text, ent.label_)
################################################################################

# There can be misses
################################################################################
text = "Upcoming iPhone X release date leaked as Apple reveals pre-orders"
doc = nlp(text)
for token in doc.ents: print(token.text, token.label_)
iphone_x = doc[1:3] # get the span for "iPhone X"
print("Missing entity:", iphone_x.text)
################################################################################

# Using the Matcher
################################################################################
# One pattern is for matching a set of *consecutive* patterns
from spacy.matcher import Matcher
doc = nlp("Upcoming iPhone X release date leaked as Apple reveals pre-orders")
matcher = Matcher(nlp.vocab)
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_X_PATTERN", [pattern])
matches = matcher(doc)
print("Matches:", [doc[start:end].text for match_id, start, end in matches])
################################################################################

# Using Matcher more complex 1
################################################################################
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)
pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]
matcher.add("IOS_VERSION_PATTERN", [pattern])
matches = matcher(doc)
print("Total matches found:", len(matches))
for match_id, start, end in matches: print("Match found:", doc[start:end].text)

################################################################################

# Using Matcher more complex 2
################################################################################
doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)
pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]
matcher.add("DOWNLOAD_THINGS_PATTERN", [pattern])
matches = matcher(doc)
print("Total matches found:", len(matches))
for match_id, start, end in matches: print("Match found:", doc[start:end].text)
################################################################################

# Using Matcher more complex 2
################################################################################
doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)
# Write a pattern for adjective plus one or two nouns
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]
matcher.add("ADJ_NOUN_PATTERN", [pattern])
matches = matcher(doc)
print("Total matches found:", len(matches))
for match_id, start, end in matches: print("Match found:", doc[start:end].text)
################################################################################
