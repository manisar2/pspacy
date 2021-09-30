# Solutions for the Free Interactive [Spacy](https://spacy.io/) 101 Course

While I took this course, I stacked the code nicely for my future reference.  

1. #### This repo has that code with appropriate chapters, sections, headings and comments.

2. #### Edits have been made for [spaCy](https://spacy.io/) v3 compatibility (this is yet to be done in online course as of 29 Sep 2021).
   This fixes the problems with `matcher.add()` and `nlp.update()` as has been asked in questions like these:
   
   [Problem with using spacy.matcher.matcher.Matcher.add() method](https://stackoverflow.com/questions/66164156/problem-with-using-spacy-matcher-matcher-matcher-add-method)  
   [How can i work with Example for nlp.update problem with spacy3.0](https://stackoverflow.com/questions/66675261/how-can-i-work-with-example-for-nlp-update-problem-with-spacy3-0)

3. #### The needed data files have been placed in the same (root) directory, and the code has been changed accordingly. This is to avoid the need of using os.path in Windows :).

The course can be started from https://spacy.io/usage/spacy-101 or https://course.spacy.io/.en/.

# What’s spaCy?

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python.

If you’re working with a lot of text, you’ll eventually want to know more about it. For example, what’s it about? What do the words mean in context? Who is doing what to whom? What companies and products are mentioned? Which texts are similar to each other?

spaCy is designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning.
