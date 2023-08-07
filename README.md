# amortized-approximation-of-induced-distribution
Brief Overview:
train_teacher_mcdp_emnist.py trains the mnist_net model on the emnist dataset to get the teacher model. 
Then in amt_approx_emnist_mmd.py, the student model approximates the teacher model by using the Dirichlet distribution. 
I’ve commented  amt_approx_emnist_mmd.py,, and these same comments and explanations are the same for the other files, 
they just use different metrics like EMD instead of Maximum Mean Discrepancy (MMD), and they use different underlying models and datasets.
