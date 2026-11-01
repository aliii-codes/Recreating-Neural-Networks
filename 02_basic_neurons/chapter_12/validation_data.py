"""
Chapter 12 - Validation Data (conceptual chapter)

This chapter has no runnable code — it is a conceptual explanation of why we
need three separate datasets and how to split them.

Key takeaways:
---------------

1. The TEST dataset must never be used to tune hyperparameters. If we tune on
   the test data, we are effectively optimizing the model against it, which is
   another way to overfit the test set itself.

2. Hyperparameter tuning should be done with a third, VALIDATION dataset:

       training data   -> used to train the model
       validation data -> used to tune hyperparameters / model selection
       testing data    -> used once, at the very end, to check generalization

3. When we cannot afford to hold out a separate validation set (not enough
   data), we have two options:

   a) Temporarily split the training data into a smaller training set and a
      validation set. Once the best hyperparameters are found, retrain on ALL
      of the training data (the test set is still untouched).

   b) k-fold CROSS-VALIDATION: split the training data into k chunks. Train on
      k-1 chunks and validate on the remaining one, rotating which chunk is used
      for validation until every chunk has been validated once. This uses all
      data for both training and validation across folds.

4. In practice we usually loop over a handful of promising hyperparameter sets
   (not all possible combinations), train each, and pick the best according to
   the validation set.
"""

print(__doc__)
