import en_core_web_sm
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
nlp = en_core_web_sm.load()
print(nlp.pipe_names)
print(nlp.pipeline)

# 6.0 Simple Components
################################################################################
from spacy.language import Language
@Language.component("length_component")
def length_component_func(doc):
    doc_length = len(doc)
    print(f"This document is {doc_length} tokens long.")
    return doc
lc = nlp.add_pipe("length_component", first=True)
print(nlp.pipe_names)
doc = nlp("This is a sentence.")
################################################################################

################################################################################
# 6.1 Complex Components
# Use the PhraseMatcher to find animal names in the document and adds the matched
    #  spans to the doc.ents
################################################################################
nlp = spacy.load("en_core_web_sm")
animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animals))
print("animal_patterns:", animal_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", animal_patterns)

@Language.component("animal_component")
def animal_component_func(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
    doc.ents = spans
    return doc
ac = nlp.add_pipe("animal_component", after="ner")
print(nlp.pipe_names)

doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])
################################################################################


################################################################################
# 6.2 Custom Extensions
    # Three Types - attribute, property and method, can be set on Doc, Token, Span
    # Attribute <= default=, Property <= getter=, Method <= method=
    # Below - custome attribute on token
################################################################################
from spacy.tokens import Token
nlp = en_core_web_sm.load()
Token.set_extension("is_country", default=False)
doc = nlp("I live in Spain.")
doc[3]._.is_country = True
print([(token.text, token._.is_country) for token in doc])
################################################################################

# 6.2.2 Custom Property on Token
################################################################################
nlp = en_core_web_sm.load()
def get_reversed(token): return token.text[::-1]
Token.set_extension("reversed", getter=get_reversed, force=True) # force => overwrite
doc = nlp("All generalizations are false, including this one.")
for token in doc: print("reversed:", token._.reversed)
################################################################################

# 6.2.3 Custom Property on Doc 
################################################################################
from spacy.tokens import Doc
nlp = en_core_web_sm.load()
def get_has_number(doc): return any(token.like_num for token in doc)
Doc.set_extension("has_number", getter=get_has_number)
doc = nlp("The museum closed for five years in 2012.")
print("has_number:", doc._.has_number)
################################################################################

# 6.2.4 Custom Method on Span
################################################################################
from spacy.tokens import Span
def to_html(span, tag): return f"<{tag}>{span.text}</{tag}>"
Span.set_extension("to_html", method=to_html)
doc = nlp("Hello world, this is a sentence.")
span = doc[0:2]
print(span._.to_html("strong"))
################################################################################

# 6.2.5 Mix of Extensions ######################################################
################################################################################
nlp = spacy.load("en_core_web_sm")
def get_wikipedia_url(span):
    # Get a Wikipedia URL if the span has one of the labels
    if span.label_ in ("PERSON", "ORG", "GPE", "LOCATION"):
        entity_text = span.text.replace(" ", "_")
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text
Span.set_extension("wikipedia_url", getter=get_wikipedia_url)
doc = nlp(
    "In over fifty years from his very first recordings right through to his "
    "last album, David Bowie was at the vanguard of contemporary culture."
)
for ent in doc.ents: print(ent.text, ent._.wikipedia_url)
################################################################################

# 6.2.6 Custom Components with Custom Extensions ###############################
    # Complete the countries_component and create a Span with the label "GPE" 
    #   (geopolitical entity) for all matches.
    # Add the component to the pipeline.
    # Register the Span extension attribute "capital" with the getter get_capital.
    # Process the text and print the entity text, entity label and entity 
    #   capital for each entity span in doc.ents.
################################################################################
import json
from spacy.lang.en import English
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy.language import Language

with open("countries.json", encoding="utf8") as f: COUNTRIES = json.loads(f.read())
with open("capitals.json", encoding="utf8") as f: CAPITALS = json.loads(f.read())

nlp = English()
matcher = PhraseMatcher(nlp.vocab)
matcher.add("COUNTRY", list(nlp.pipe(COUNTRIES)))

@Language.component("countries_component")
def countries_component(doc):
    # Create an entity Span with the label "GPE" for all matches
    matches = matcher(doc)
    doc.ents = [Span(doc, start, end, label="GPE") for match_id, start, end in matches]
    return doc

nlp.add_pipe("countries_component", last=True) # add the component to the pipeline
print(nlp.pipe_names)

# Getter that looks up the span text in the dictionary of country capitals
get_capital = lambda span: CAPITALS.get(span.text)

# Register the Span extension attribute "capital" with the getter get_capital
Span.set_extension("capital", getter=get_capital)

# Process the text and print the entity text, label and capital attributes
doc = nlp("Czech Republic may help Slovakia protect its airspace")
print([(ent.text, ent.label_, ent._.capital) for ent in doc.ents])
################################################################################

################################################################################
# 6.3 Processing Streams
################################################################################
nlp = en_core_web_sm.load()
with open("tweets.json", encoding="utf8") as f: TEXTS = json.loads(f.read())
​
# Process the texts and print the adjectives
# BAD
for text in TEXTS:
    doc = nlp(text)
    print([token.text for token in doc if token.pos_ == "ADJ"])

# GOOD
docs = list(nlp.pipe(TEXTS))
for doc in docs: print([token.text for token in doc if token.pos_ == "ADJ"])

entities = [doc.ents for doc in list(docs)]
print(*entities)

people = ["David Bowie", "Angela Merkel", "Lady Gaga"]
# Create a list of patterns for the PhraseMatcher
# patterns = [nlp(person) for person in people] # BAD
patterns = list(nlp.pipe(people))

################################################################################

################################################################################
# 6.4 Processing Data with Contexts ############################################
    # Using custom attributes to add author and book meta information to quotes.
    # A list of [text, context] examples is available as the variable DATA. 
    #   The texts are quotes from famous books, and the contexts dictionaries with the keys "author" and "book".
    # - Use the set_extension method to register the custom attributes "author" 
    #   and "book" on the Doc, which default to None.
    # - Process the [text, context] pairs in DATA using nlp.pipe with as_tuples=True.
    # - Overwrite the doc._.book and doc._.author with the respective info passed 
    #   in as the context.
################################################################################
import json
from spacy.lang.en import English
from spacy.tokens import Doc

with open("bookquotes.json", encoding="utf8") as f: DATA = json.loads(f.read())
nlp = English()
Doc.set_extension("author", default=None)
Doc.set_extension("book", default=None)
for doc, context in nlp.pipe(DATA, as_tuples=True):
    # Set the doc._.book and doc._.author attributes from the context
    doc._.book = context["book"]
    doc._.author = context["author"]
    # Print the text and custom attribute data
    print(f"{doc.text}\n — '{doc._.book}' by {doc._.author}\n")
################################################################################

################################################################################
# 6.4 Selective Processing #####################################################
################################################################################
import spacy
nlp = spacy.load("en_core_web_sm")
text = (
    "Chick-fil-A is an American fast food restaurant chain headquartered in "
    "the city of College Park, Georgia, specializing in chicken sandwiches."
)
# doc = nlp(text) # this runs the components immediated after making the doc
doc = nlp.make_doc(text) # this won't run the components except tokenizer (why tokenizer? TBC)
print([token.text for token in doc])
################################################################################

# 6.4.2
################################################################################
with nlp.disable_pipes("tagger", "parser"):
    doc = nlp(text)
    print(doc.ents)
################################################################################

