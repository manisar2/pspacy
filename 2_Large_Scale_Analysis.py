import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# Tokens have attrs like pos_, dep_ etc. that depend on the context
    # Lexemes seem to be independent
    # New words, if needed, can be added to .strings by
    #   1. processing a text
    #   2. looking up the string (TBC)
    #   3. using the same vocab (TBC)

# 1.0
################################################################################
doc = nlp("I have a cat")
cat_hash = nlp.vocab.strings["cat"]
print(cat_hash)

cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)
################################################################################

# 2.0
################################################################################
doc = nlp("David Bowie is a PERSON")
person_hash = nlp.vocab.strings["WOW"]
print(person_hash)

# Look up the person_hash to get the string
person_string = nlp.vocab.strings[person_hash]
print(person_string)
################################################################################

# 3.0 Data Structures: Doc, Span and Token Subtleties
################################################################################
from spacy.tokens import Doc
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
################################################################################

# 3.0.1
################################################################################
words = ["Go", ",", "get", "started", "!"]
spaces = [False, True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
################################################################################

# 3.0.2
################################################################################
words = ["Oh", ",", "really", "?", "!"]
spaces =[False, True, False, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
################################################################################

################################################################################
# 3.0.3 Adding a span to doc.ents
################################################################################
from spacy.tokens import Span
words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

span = Span(doc, 2, 4, label="PERSON")
doc.ents = [span]

print([(ent.text, ent.label_) for ent in doc.ents])
################################################################################

################################################################################
# 3.0.4 Data Structures Best Practices
################################################################################
doc = nlp("Berlin looks like a nice city")
token_texts = [token.text for token in doc]
pos_tags = [token.pos_ for token in doc]

for token in doc:
    if token.pos_ == 'PROPN' and doc[token.i+1].pos_ == "VERB":
        print("Found proper noun before a verb:", token.text)

################################################################################

################################################################################
# 4.0 Similarity - Doc., Span., Token.
################################################################################
import en_core_web_md
nlp = en_core_web_md.load()
doc = nlp("Two bananas in pyjamas")
bananas_vector = doc[1].vector
print(bananas_vector)
################################################################################

# 4.0.1
################################################################################
doc = nlp("TV and books")
token1, token2 = doc[0], doc[2]
similarity = token1.similarity(token2)
print(similarity)
################################################################################

# 4.0.2
################################################################################
doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")
# Create spans for "great restaurant" and "really nice bar"
span1 = doc[3:5]
span2 = doc[12:15]
similarity = span1.similarity(span2)
print(similarity)
################################################################################

################################################################################
# 5.0 Debugging Patterns
from spacy.matcher import Matcher
# pattern1 matches all case-insensitive mentions of "Amazon" plus a title-cased proper noun.
# pattern2 matches all case-insensitive mentions of "ad-free", plus the following noun.
doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)
pattern1 = [{"LOWER": "amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
pattern2 = [{"LOWER": "ad"}, {"TEXT": "-"}, {"LOWER": "free"}, {"POS": "NOUN"}]
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", [pattern1])
matcher.add("PATTERN2", [pattern2])
for match_id, start, end in matcher(doc):
    print(doc.vocab.strings[match_id], doc[start:end].text)
################################################################################

################################################################################
# 6.0 Efficient Phrase Matching
    # Sometimes it’s more efficient to match exact strings instead of writing patterns 
    # describing the individual tokens
################################################################################
import json
with open("countries.json", encoding="utf8") as f: COUNTRIES = json.loads(f.read())
doc = nlp("Czech Republic may help Slovakia protect its airspace")
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", [*patterns])
matches = matcher(doc)
print([doc[start:end] for match_id, start, end in matches])
################################################################################

# 6.0.1 Extract Countries and Relationships
    # Iterate over the matches and create a Span with the label "GPE" (geopolitical entity).
    # Overwrite the entities in doc.ents and add the matched span.
    # Get the matched span’s root head token.
    # Print the text of the head token and the span.
################################################################################
import json
with open("countries.json", encoding="utf8") as f: COUNTRIES = json.loads(f.read())
with open("country_text.txt", encoding="utf8") as f: TEXT = f.read()
matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", [*patterns])
doc = nlp(TEXT)
doc.ents = []
for match_id, start, end in matcher(doc):
    span = Span(doc, start, end, label="GPE")
    doc.ents = list(doc.ents) + [span]
    span_root_head = span.root.head
    print(span_root_head.text, "-->", span.text)
print([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"])    
################################################################################
