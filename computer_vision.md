# Computer Vision Papers

## DINO [https://arxiv.org/pdf/2104.14294]
* DINO: self-**di**stillation with **no** labels
* Facebook trained a vision transformer backbone with self supervised learning
  * reminder: unsupervised learning (e.g clustering) is about learning patterns in the data without a predefined task
  * self supervised learning however is a form of supervised learning where the labels are generated from the data itself (i.e no human labelling needed)
* "In this paper, we question whether the muted success of
Transformers in vision can be explained by the use of super-
vision in their pretraining. Our motivation is that one of the
main ingredients for the success of Transformers in NLP was
the use of self-supervised pretraining"
* "self-supervised pretraining objectives use the words
in a sentence to create pretext tasks that provide a richer
learning signal than the supervised objective of predicting
a single label per sentence. Similarly, in images, image-
level supervision often reduces the rich visual information
contained in an image to a single concept selected from a
predefined set of a few thousand categories of objects"
