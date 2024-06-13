# 19. Learning with Fewer Labeled Examples

Many ML models often have many more parameters than we have labeled training examples. A ResNet CNN with 50 layers will have 25 millions parameters, and transformer models can be even bigger!

These models are slow to train and may easily overfit. This is particularly a problem when you don’t have a large labeled training set.

We discuss ways to tackle this issue, beyond the generic regularization technique like early stopping, weight decay and dropout.