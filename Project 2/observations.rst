Things
======

Normalization
-------------

- Normalizing the whole image first is fast, but terrible.
- Normalizing rows of height patch-size is pretty fast, and 
  produces decent results
- Normalizing each patch is mega-slow, but produces the best 
  outcome.
