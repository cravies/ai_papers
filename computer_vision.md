# Computer Vision Papers

## DINO [https://arxiv.org/pdf/2104.14294] ✅📜
* DINO: self-**di**stillation with **no** labels
* Facebook trained a vision transformer backbone with self supervised learning
  * Unsupervised learning is about learning patterns in the data without any labels or supervision. Example: clustering
  * Self supervised learning is a form of supervised learning where the labels are generated from the data itself (i.e no human labelling needed). To do this you have to define a "pre-text task"
  * DINO's "pre-text task" is learning to map global image representations (from the teacher) to the local representations of the student. In this way 
* _"In this paper, we question whether the muted success of
Transformers in vision can be explained by the use of super-
vision in their pretraining. Our motivation is that one of the
main ingredients for the success of Transformers in NLP was
the use of self-supervised pretraining"_
* _"self-supervised pretraining objectives use the words
in a sentence to create pretext tasks that provide a richer
learning signal than the supervised objective of predicting
a single label per sentence. Similarly, in images, image-
level supervision often reduces the rich visual information
contained in an image to a single concept selected from a
predefined set of a few thousand categories of objects"_
* _"knowledge distillation ... (is the process of) training a small network to mimic the
output of a larger network to compress it"_
* _"Our work extends knowledge distillation to the case
where no labels are available."_
* Usually, knowledge distillation means one pretrained (or labeled) model (teacher) teaches another model (student); 
* It's expensive to directly train a teacher model when labels aren't available
* DINO instead applies knowledge distillation without labels, training teacher and student simultaneously.
* In DINO, the "teacher" is updated by averaging the student's weights over time.
* _" the teacher in
codistillation is distilling from the student, while it is
updated with an average of the student in our work."_
* Averaging the student's weights to form the teacher creates a stable, low-noise reference, that acts as a "memory", storing consistent patterns
### Normal Knowledge Distillation Paradigm
* Train student network $$g_{θ_{s}}$$ to match teacher output $$g_{θ_{t}}$$. They both model probability distributions $$p_{s}(x)$$ and $$p_{t}(x)$$: 
* I.e minimise cross entropy of
```math
\text{min} (\theta_{s})[-p_{t}(x) \text{log}(p_{s}(x))]
```
### DINO Paradigm
* For every image in the dataset, generate a set $V$ of augmentations. 
* This set contains two full size augmentations $x_{1}^{g},x_{2}^{g}$ and several smaller (cropped) image augmentations
* All augmentation are passed to the student, only global ones are passed to the teacher
* This encourages the student to learn to recognize global patterns from smaller crops. I.e the student gains stronger feature representations because it must generalise from the particular.
* The loss function is then
```math
\min_{\theta_s} \sum_{x \in \{x_1^g, x_2^g\}} \sum_{x' \in V, x' \neq x} H(P_t(x), P_s(x'))
```
* Both networks share the same architecture g with different sets of parameters
* The teachers weights are a exponential moving average of the student's weights
* The architecture is a backbone (resnet or vit) with a projection head (fully connected 3 layer MLP)
#### Avoiding model collapse
* Model collapse is when the network learns to just output a shortcut or hack instead of learning features that will generalise
* DINO avoids this by sharpening and centering the output distribution
* Sharpening avoids uniform distribution collapse, centering avoids single dimension collapse
#### Summing it all up
* The teacher network sees only global crops, creating stable, high-level semantic embeddings. The student network sees local crops and learns to predict the teacher's global-level embeddings from local crops, forming a supervised task ("local views must match global views") without external labels. Thus this is a form of supervised learning, and it produces features that are superior at generalising
