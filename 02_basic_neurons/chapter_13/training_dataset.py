"""
Chapter 13 - Training Dataset (conceptual chapter)

This chapter has no runnable code — it covers how to prepare the training
dataset (a process called "preprocessing").

Key takeaways:
---------------

1. Neural networks usually perform best on data scaled to a range of 0 to 1,
   or (preferably) -1 to 1. Centering data on 0 helps training by attenuating
   weight bias in one direction.

2. Why scaling matters:
   - Many activation functions behave well within these ranges (softmax and
     sigmoid output 0..1, tanh outputs -1..1).
   - Networks do lots of multiplication; large values can cause floating-point
     overflow/instability and make weights grow too fast.

3. Simple linear scaling example for image data (0..255):

       X = (X - 127.5) / 127.5   # maps 0..255  ->  -1..1

4. CRITICAL: apply the SAME scaling to training, validation, testing, and any
   future prediction data. Prepare a scaler once on the training data and reuse
   that same instance everywhere (and save it alongside the model).

5. DATA AUGMENTATION: when we have too few samples, we can create new training
   samples by applying realistic transformations (rotate, crop, shift, etc.).
   Only use augmentations similar to variations we would see in reality.

6. How many samples do we need? There is no single answer. Roughly:
   - a few hundred per class for very simple problems
   - a few thousand to tens of thousands per class for most real problems
   - millions for complex tasks like natural language
"""

print(__doc__)
