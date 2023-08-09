# Open-Set-Recognition-With-Contrastive-Learning

## Introduction:
In the realm of machine learning and artificial intelligence, we frequently strive to 
improve the predictive prowess of our models. These endeavors oFen concentrate 
on tasks within the confines of a closed world, wherein the models are trained and 
evaluated using the same set of classes. However, the real world offers an open set 
environment, where we encounter numerous classes that a model may not have 
been trained on. Our project aims to delve into the realm of Open Set Recognition 
(OSR), an increasingly prevalent yet understudied area of deep learning.

Open Set Recognition is an intriguing problem that combines aspects of classification 
and anomaly detection. Unlike traditional classification problems, where a model 
learns and is tested on a known set of classes, OSR presents a more realistic and 
challenging scenario. The model, trained on known classes, may confront unseen 
classes during testing. This differs significantly from the closed-set scenario, and it 
provides a fascinating avenue for exploring the robustness of our models and the 
sophistication of their decision boundaries.

The importance of this research domain cannot be overstated, given its relevance to 
many real-world applications. For example, in computer vision systems for 
autonomous vehicles, encountering an unknown object is more a rule than an 
exception. Similarly, in the field of medical imaging, a model might be presented 
with diseases that were not part of the training data.

In this project, we opt to use the MNIST dataset, a classic in the machine learning 
community, as our closed-set classes. Our goal, however, is not limited to correctly 
classifying these MNIST examples, a task that has been thoroughly explored. We aim 
to push the boundaries by requiring the model to be able to flag unseen classes 
during test time as 'Unknown'. This introduces an additional layer of complexity to 
the task.

To put it concretely, during testing, the model will not only receive examples from 
the MNIST test dataset but also examples from potentially unrelated datasets. These 
could be anything from leZers from an alphabet, images of animals, or perhaps even 
more abstract concepts. This ensures that the model is not merely recognizing digits 
but also distinguishing them from other classes of images.

Our focus on OSR is to explore the breadth of what can be achieved with existing 
deep learning models when faced with unknown classes. It's about creating a more 
adaptable, robust, and accurate AI that is more in tune with the open world's 
unpredictability and diversity. This project, therefore, not only serves as an exciting 
opportunity for innovation and learning but also a step forward in our ongoing 
pursuit of creating AI models that can effectively navigate the real world.

The evaluation of our project will be carried out on two major criteria:
• The model's accuracy in correctly classifying the 10 digits of the MNIST dataset.
• The model's proficiency in correctly flagging unseen classes as 'Unknown'.

To create an effective OSR model, we will aim to meet the following objectives:
1. Mitigate the decrease in accuracy relative to our baseline model as much as 
possible.
2. Limit the computational overhead of our OSR classifier, ensuring our solution 
is both efficient and practical.
3. Distinguish between low-confidence samples (e.g., poorly wriZen digits) and 
out-of-distribution ones (e.g., an image of a bird), an essential aspect for 
preventing false 'Unknown' classifications.
4. Develop the capability to correctly classify unseen classes as 'Unknown', a 
core requirement for any OSR model.

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/d1fa4eed-c0d5-4ff7-bf6a-4d9fdc14bfff)

## Related Work
Open Set Recognition has its roots in the works of Scheirer. who first coined the 
term, introducing an approach based on SVMs. Later, Bendale and Boult proposed 
OpenMax, an extension of soFmax for OSR, that estimates the probability of 
unknown classes. Neural-based methods like Out-of-distribution (OOD) detection 
and Generative models have also been employed for OSR. More recent works like G-OpenMax combine generative models with OpenMax. Lastly, Meta-recognition 
techniques, leveraging metadata, have shown promise in OSR.

## Our Methodology: A High-Level Overview:

Our model uses a deep learning approach, specifically a Convolu=onal Neural 
Network (CNN), as the backbone to handle image data and extract meaningful 
representa=ons. The CNN architecture is used due to its ability to effec=vely process 
high-dimensional data and capture local correla=ons and transla=on invariances in 
images.

The CNN processes the input image and extracts high-level features. The final layer 
of the CNN (excluding the soFmax layer) outputs these feature representa=ons, 
oFen referred to as latent vectors.

The objec=ve of training this model is to learn a discrimina=ve space where the indistribu=on samples are =ghtly clustered according to their classes and the out-ofdistribu=on samples are separated. In order to achieve this, the training employs a combination of a soFmax cross-entropy loss and a contras=ve loss - InfoNCE Loss.

The cross-entropy loss ensures that the CNN learns to correctly classify the indistribu=on samples. On the other hand, contras=ve loss plays a pivotal role in our 
model to ensure the crea=on of meaningful representa=ons of our data, which is 
crucial for addressing the OSR problem. It operates on the principle that similar data 
points should be closer in the latent space and dissimilar ones should be farther 
apart.

Here, we create 'positive pairs' and 'negative pairs'. Positive pairs are created from 
an original image and its augmented version, as both belong to the same class. 
Nega=ve pairs, on the other hand, consist of an original image and a randomly 
chosen different image from another class. The aim is to minimize the distance 
between posi=ve pairs and maximize the distance between nega=ve pairs in the 
representa=on space.

The contras=ve loss ensures that different augmenta=ons of an image, which belong 
to the same class, get mapped close to each other in the latent space. At the same 
=me, it ensures that images from different classes, which should be dissimilar, get 
mapped farther apart.

This idea of pushing similar things closer and dissimilar ones apart shapes the 
learned latent space into clusters of similar class examples and significantly 
separated clusters of different classes. This significantly enhances the discriminatory 
power of the model.

In order to recognize out-of-distribu=on samples during inference, decision 
boundaries in the latent space are defined for each class based on the in-distribu=on 
data.

Decision boundaries in the latent space play a fundamental role in iden=fying out-ofdistribu=on samples. For each class, the latent space will contain a cluster of points, 
each represen=ng an image of that class. We calculate the mean of this cluster and 
use it as a representa=on for that class.

However, not all points within a class cluster are equidistant from the mean. Hence, 
we set a distance threshold (e.g., 97%), such that a large majority of the indistribu=on samples' distances from the class mean are within this threshold. This 
creates a hypersphere around the mean latent vector of each class, effec=vely 
delinea=ng the decision boundaries.

An unseen sample is classified as belonging to a par=cular class if its latent 
representa=on falls within the decision boundary of that class. If the sample doesn't
fall within the decision boundary of any known class, it is classified as an out-of-distribution sample.
In this way, the decision boundaries work in tandem with the contras=ve loss. The 
contras=ve loss ensures a well-clustered latent representa=on space and decision 
boundaries use this space to segregate in-distribu=on and out-of-distribu=on 
samples.

This intricate combina=on of contras=ve loss and decision boundaries in the latent 
space is the cornerstone of our approach to the OSR problem. It enables our model 
to recognize whether a given sample is within the realm of its training (indistribu=on) or beyond it (out-of-distribu=on), thus providing a robust solu=on to 
the challenging OSR problem.

Here is some decision boundary visualiza=on:
![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/bd9affa3-5869-4d75-a26d-32b2bf3bca6c)

## Experiments

Our research journey involved mul=ple stages of experimenta=on, during which we 
explored a range of architectures and approaches. The focus was on leveraging 
latent embeddings and capitalizing on the power of representa=onal learning to 
solve the out-of-distribu=on sample recogni=on (OSR) problem. Here, we recount 
some of the significant milestones in our journey:

1. Autoencoder with a ReconstrucBon Loss Threshold: Ini=ally, we employed an 
autoencoder, u=lizing a reconstruc=on loss threshold to separate out-of-distribu=on 
samples. The autoencoder architecture was designed to learn a compact, efficient 
latent embedding of the input data. However, relying solely on a reconstruc=on loss 
threshold, while simple, was not sufficiently discrimina=ve to differen=ate between 
in-distribu=on and out-of-distribu=on samples effec=vely.

2. ConvoluBonal Neural Network (CNN): Aiming for a stronger representa=on 
learning approach, we experimented with a basic Convolu=onal Neural Network 
(CNN). Although the CNN model performed well on the MNIST in-distribu=on 
dataset, it struggled to generalize and accurately iden=fy out-of-distribu=on 
samples, likely due to the inability to form well-separated latent embeddings for 
each class.

3. VariaBonal Autoencoder (VAE): We then turned to Varia=onal Autoencoders 
(VAEs), hoping to harness their probabilis=c nature and more expressive latent 
space. The VAE did provide more structure to the latent embeddings, but it s=ll fell 
short of expecta=ons. The limita=on arose from the fact that the VAEs' primary focus 
is on data reconstruc=on rather than learning discrimina=ve features for 
classifica=on.

4. Fully-Connected Network with ContrasBve Loss: A more successful aZempt 
involved using a fully connected network coupled with contras=ve loss. The 
combina=on allowed us to focus on learning robust latent embeddings that 
emphasize inter-class differences and intra-class similari=es. However, it was not 
sufficient by itself to solve the OSR problem due to the absence of spa=al correla=on 
considera=on, which is essen=al when working with image data.

5. RegularizaBon Techniques (L1/L2 RegularizaBon): We also experimented with 
adding L1 and L2 regulariza=on to the exis=ng architecture, seeking to prevent 
overfiing and improve generaliza=on. While this enhanced the performance on the 
in-distribu=on MNIST dataset quite significantly, it was detrimental for the OOD 
recogni=on, indica=ng a possible trade-off between in-distribu=on performance and 
OOD detec=on capability.

6. Exploring Other Distance Metrics: Instead of the tradi=onal Euclidean distance 
used in the contras=ve loss, we tested both L1 distance and cosine similarity. The 
idea was to ascertain if a change in the distance metric might yield more 
discrimina=ve latent embeddings. The results were mixed, with minor improvements 
but nothing groundbreaking.

7. Orthogonal Loss Component Using Cosine Similarity: Finally, we aZempted to 
introduce orthogonality in the learned representa=ons by adding an orthogonal loss 
component using cosine similarity. This technique was aimed at enforcing 
orthogonality between the learned feature vectors, thereby promo=ng diversity and 
separa=on in the latent space. Although this led to some improvements, it wasn't 
sufficient to deliver a comprehensive solu=on for OSR.

In conclusion, each of these experiments provided us with invaluable insights, which 
led to the crea=on of our current architecture. The final model successfully balances 
the need for high in-distribu=on accuracy, effec=ve OOD detec=on, and efficient 
latent representa=on. It accomplishes this through the synergy of a deep 
convolu=onal architecture, contras=ve learning, and an appropriately chosen 
threshold, all working in tandem to create well-separated, discrimina=ve 
embeddings in the latent space.

## Model Architecture:

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/a616e083-2f64-477b-987e-77565df7bff5)

MNIST images are grayscale and have a resolu=on of 28x28 pixels. So, the input 
tensor x starts with the shape [bs, 1, 28, 28]. Here's how the shape of x evolves as it 
passes through our model:
1. Input Layer: The input x has a shape of [bs, 1, 28, 28].
2. ConvoluBonal Layer 1 (conv1): AFer passing through the first convolu=onal layer, 
ReLU , and max pooling layer we get the shape of [bs, 16, 14, 14].
3. ConvoluBonal Layer 2 (conv2): AFer the second convolu=onal layer, ReLU and 
max pooling layer the shape is [bs, 32, 7, 7].
4. Fla[ening: At this point, the 3D output tensor is flaZened to a 1D tensor for each 
image in the batch. This leads to a shape of [bs, 1568].
5. Fully Connected Layer 1 (fc1): This layer transforms x from [bs, 1568] to [bs, 15]
then we ac=vate ReLU func=on.
6. Fully Connected Layer 2 (fc2): This layer further transforms x from [batch_size, 15] 
to [batch_size, 10].
7. LogSo\max AcBvaBon: AFer this layer, the shape of x is s=ll [bs, 10]. Then, there 
is a concatena=on opera=on that adds an addi=onal dimension, which results in a 
final output shape of [bs, 11]

We have extensively researched and experimented with numerous deep-learning 
architectures for our applica=on, striving to ensure the most efficient and accurate 
performance. The architecture that has demonstrated superior performance is the 
described Convolu=onal Neural Network (CNN) model. The core structure of this 
model consists of two convolu=onal layers followed by two fully connected layers, 
with an addi=onal opera=on that uses a set of means and thresholds to adjust the 
output dynamically.

Nevertheless, it is important to men=on that our choice was not arbitrary but was a 
result of an extensive and systema=c evalua=on process that included several other 
architectures and techniques. Here are some of the approaches that we 
experimented with, and although they presented valuable insights, they fell short 
compared to the selected CNN model:

1. Shallow Networks: We started our journey with rela=vely simple and shallow 
architectures. Although these models were computa=onally less demanding and 
easy to interpret, they failed to capture the complex features necessary for high 
accuracy in our applica=on.

3. Dense Neural Networks: These models were an improvement over the shallow 
ones, capturing more complex representa=ons. However, they were not op=mal due 
to their inability to handle the spa=al informa=on in the image data effec=vely.

5. Other CNN Variants: We experimented with different configura=ons of 
Convolu=onal Neural Networks, varying the number of layers, kernel sizes, and other 
hyperparameters. Although some of them performed reasonably well, none could 
outperform our current model.

7. Pretrained Models: Leveraging the power of transfer learning, we also used 
pretrained models like ResNet, VGG, and Incep=on. While these models performed 
well on tasks with larger and more complex images, they were less efficient and 
somewhat overkill for our use-case of MNIST digit classifica=on.

9. Advanced Techniques: We incorporated advanced techniques such as Batch 
Normaliza=on, Dropout, and different types of regulariza=on into our models. These 
techniques, while helping with generaliza=on and preven=ng overfiing to some 
extent, didn't enhance the performance as much as our current architecture did.

11. Various AcBvaBon FuncBons: We tried replacing ReLU with other ac=va=on 
func=ons such as LeakyReLU, ELU, and SELU. However, these modifica=ons didn't 
provide substan=al improvements in our case.

AFer rigorous tes=ng and op=miza=on, the described CNN model yielded the best 
results in terms of both performance and computa=onal efficiency. Our experience 
further reaffirmed the necessity of tailor-made solu=ons for each specific task in the 
realm of deep learning. The current architecture's robustness and adaptability make 
it an excellent fit for our applica=on's specific requirements.


## Data Augmentation
The data augmenta=on method that we used in our model in order to create 
‘posi=ve’ examples consists of several image transforma=ons:

1. Gaussian Blur: Gaussian blur is a smoothing opera=on applied to images. It 
reduces the level of detail and noise in the image by averaging the pixel values in the 
vicinity of each pixel with a Gaussian kernel. The strength of the blur is controlled by 
the standard devia=on of the Gaussian distribu=on.

3. Random Affine TransformaBons: Affine transforma=ons are geometric opera=ons 
that include rota=on, transla=on (shiFing), scaling (resizing), and shearing 
(distor=on). In this data augmenta=on method, random values are applied to control 
the degree of rota=on, transla=on, and scaling. The purpose of these 
transforma=ons is to introduce variability in the posi=on and size of objects in the 
image

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/d05c2e8f-f719-42be-bf72-2483f56ee039)


## InfoNCE Loss
InfoNCE, where NCE stands for Noise-Contras=ve Es=ma=on, is a type of 
contras=ve loss func=on used for self-supervised learning.

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/9b68bd65-e81c-4aac-8752-4df4e4eded82)

In the provided graph, a prominent convergence trend becomes apparent around 
the 6th epoch. The model achieves a notably small loss value, showcasing its success 
in classifying MNIST images effec=vely. Importantly, the marginal difference 
between the loss on the valida=on set and the training set indicates the absence of 
overfiing. From the seventh epoch beginning, we observed an upward trend in 
valida=on loss (as we can see in the graph above), contradic=ng the an=cipated 
downward trajectory in the training loss.
Moreover, the test data reflects an impressive baseline accuracy percentage of 
98.06%, underscoring the excellent performance of our model in MNIST 
classifica=on.

We collected 4 images that the classifier didn’t classify well -
![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/a98b592a-c554-412e-a003-70be1b7ab463)

It can be easily noticed that even a human eye may have difficulty in classifying these 
images properly, so it is not surprising that our classifier also failed to recognize 
them, which is quite encouraging overall.

## Evaluation Results On The Combined Data Set

The presented results are indica=ve of the performance of our deep learning model 
for the Out-of-Sample Recogni=on (OSR) problem using a combined dataset of 
MNIST and a CIFRA10 - the Out-of-Distribu=on (OOD) dataset.

The results are shown for varying threshold levels in the decision boundary within 
the latent space, as discussed earlier. This threshold controls the hypersphere's size 
around the mean latent vector of each class, within which an unseen sample's latent 
representa=on is classified as belonging to that class.

In our context, an increase in the threshold value results in a larger hypersphere. 
This means the model becomes more inclusive towards the samples it accepts as 
belonging to the in-distribu=on (MNIST) classes. Conversely, a smaller threshold 
means a smaller hypersphere, which makes the model more conserva=ve, and it 
becomes stricter in classifying samples as belonging to the in-distribu=on classes.

In the provided results:
- MNIST Accuracy: Represents the model's accuracy on in-distribu=on samples from 
the MNIST dataset.
- OOD Accuracy: Represents the model's accuracy in iden=fying out-of-distribu=on 
samples.
- Total Accuracy: Represents the model's overall accuracy on both in-distribu=on and 
out-of-distribu=on samples.

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/72c5f1b5-56aa-48dd-b746-e06512f4ae42)

Let's analyze the data.

We see a trend where increasing the threshold improves the MNIST accuracy but 
decreases the OOD accuracy. This makes sense since a larger threshold allows more 
samples to be classified as in-distribu=on, improving in-distribu=on accuracy but 
reducing the model's ability to correctly iden=fy OOD samples.

At a threshold of 0.8%, the model has an MNIST Accuracy of 80.15% and an OOD 
Accuracy of 99.89%. It means it's stringent in labeling samples as in-distribu=on, 
resul=ng in a high OOD Accuracy but a lower MNIST Accuracy.

As we increase the threshold, the MNIST Accuracy increases, showing that the model 
becomes more accep=ng of in-distribu=on samples. However, this loosening of the 
decision boundaries also causes more OOD samples to be misclassified as indistribu=on, leading to a decrease in OOD Accuracy.
We reach a peak in Total Accuracy at a threshold of 0.95%, with an MNIST accuracy 
of 94.36%, an OOD accuracy of 97.99%, and a total accuracy of 96.17%. Beyond this 
point, the trade-off between MNIST Accuracy and OOD Accuracy starts to hurt the 
Total Accuracy.

By the =me we reach a threshold of 0.99%, the OOD Accuracy has dropped 
significantly to 83.43%, while MNIST Accuracy is at 97.00%, and Total Accuracy has 
fallen to 90.22%. This suggests that the model is misclassifying a considerable 
number of OOD samples as in-distribu=on, showing the limita=on of seing the 
threshold too high.

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/a1c38766-161a-4723-ae47-49ee3a42e607)

The above analysis of thresholds was performed with the model trained over 5 
epochs, aiming to demonstrate the inherent trade-off between accuracy on indistribu=on (MNIST) data and out-of-distribu=on (OOD) data. This is essen=ally a 
process of balancing the model's ability to accurately classify MNIST samples while 
also effec=vely dis=nguishing OOD samples.

However, the importance of extended training in augmen=ng the model's 
performance should not be underes=mated. With addi=onal training—for instance, 
an extra 5 epochs—we've seen a marked improvement in our model's overall 
performance, even at higher thresholds.

When we expanded training to a total of 10 epochs at a 98% threshold, we got the 
best performance MNIST accuracy soar to 97.08% and OOD accuracy reach an 
impressive 99.24%, resul=ng in a total accuracy of 98.16%, a significant 
enhancement compared to results seen at the 5-epoch mark.

This finding emphasizes the model's ability to con=nually adapt and improve. The 
ini=al epochs are pivotal in laying a solid founda=on, upon which addi=onal epochs 
allow the model to further refine its understanding and decision boundaries.

Moreover, extended training has another cri=cal impact—it enhances the model's 
ability to disentangle the OOD samples from the MNIST classes on the latent space. 
Essen=ally, as the model con=nues to train over more epochs, it is beZer able to 
carve out dis=nct decision boundaries in the latent space, resul=ng in clearer 
delinea=on between OOD and MNIST classes. This aids in both the accurate 
classifica=on of MNIST instances and the effec=ve iden=fica=on and separa=on of 
OOD instances, thereby further boos=ng the performance across both domains.

In conclusion, our research on Out-of-Distribu7on Sample Recogni7on reveals 
a cri7cal balance between recognizing in-distribu7on (MNIST) and out-ofdistribu7on data, influenced by the choice of threshold. Extended training can 
notably improve this balance, yielding superior model performance and beJer 
segrega7on of OOD samples from MNIST data in the latent space.

## Seed Robustness
![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/c19b88e0-cb2b-4819-a1aa-1730e6ea0d25)

The stability of deep learning models across different random seed ini=aliza=ons is a 
crucial facet of their robustness. Deep learning models, including ours, are stochas=c 
in nature due to the random ini=aliza=on of weights and the randomness involved in 
methods such as minibatch selec=on during stochas=c gradient descent. This means 
that every training run could poten=ally yield different results due to these random 
factors.

If a model shows significantly different results with different seed values, it might 
indicate overfiing or the chance-based discovery of specific solu=ons during 
training. Hence, it's important to examine the model's performance across different 
seed ini=aliza=ons to ensure the model's effec=veness is not a result of fortunate 
weight ini=aliza=ons or a par=cular training sample order.

In our case, the experiment has been conducted across five different seeds, ranging 
from 0 to 4. The consistent performance across these varying seeds demonstrates 
the robustness of our model. This indicates that the performance we're observing is 
likely due to the effec=veness of the model architecture, the learning algorithm, and 
the contras=ve loss func=on, rather than a result of chance or par=cular 
ini=aliza=on.

Therefore, we can confidently state that our model's performance and its proficiency 
at solving the Out-of-Sample Recogni=on (OSR) problem is robust to changes in 
random seed ini=aliza=on. This enhances the credibility of our research and the 
dependability of our model in a produc=on environment, where consistency and 
stability are paramount.

## Accuracy for different batch sizes

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/c39f34f5-3780-455c-b250-ccaacf19b30a)

The batch size plays an influen=al role beyond model performance; it also impacts 
the number of nega=ve samples we generate. Larger batches naturally result in a 
more diverse set of nega=ve pairs per epoch, improving the capacity of our model to 
dis=nguish between posi=ve and nega=ve pairs in the latent space. This enhanced 
understanding of contrasts between classes significantly contributes to our model's 
OOD detec=on proficiency. Therefore, a well-tuned batch size strikes a balance 
between computa=onal efficiency and the model's ability to discern complex 
decision boundaries in high-dimensional space.

## Latent Dim compare and TSNE visualization
![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/67eb6bfa-1bbe-40b6-9d72-beb6487799cd)

In our quest to understand the effects of varying latent dimensions on our model's 
ability to dis=nguish between in-distribu=on (MNIST) and out-of-distribu=on (OOD) 
samples, we experimented with four different latent dimensions: 10, 15, 50, and 
100.

![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/8cb9e25b-e4d3-478b-9a78-d4b9e0b317bc)

The results are quite instruc=ve:

Latent Dimension = 10: The combined test set achieved an MNIST accuracy of 
94.63% and an OOD accuracy of 86.74%, culmina=ng in a total accuracy of 90.69%. 
However, compared to the baseline MNIST test set accuracy of 96.79%, it is evident 
that the model, with a latent dimension of 10, is somewhat less successful in 
dis=nguishing OOD instances. This could be due to a lower dimensional space which 
may not adequately capture the complexity of the data. This hypothesis is 
corroborated by the t-SNE visualiza=on, where the OOD data points appear more 
sparse, indica=ng less cohesion within the OOD cluster.

Latent Dimension = 15: Here we see a significant improvement. The combined test 
set yielded an MNIST accuracy of 94.36% and an OOD accuracy of 97.99%, resul=ng 
in a total accuracy of 96.17%. This is even beZer than the baseline MNIST test set 
accuracy of 98.31%. The t-SNE visualiza=on also shows a more compact OOD cluster 
compared to the previous scenario, indica=ng that a slightly higher latent dimension 
helps the model to beZer recognize and separate OOD instances.

Latent Dimension = 50: This scenario yields mixed results. While the MNIST accuracy 
for the combined test set remains rela=vely high at 94.04%, the OOD accuracy drops 
significantly to 73.72%, leading to a total accuracy of 83.88%. This is substan=ally 
lower than the baseline MNIST test set accuracy of 96.73%. In the t-SNE visualiza=on,
we observe a class overlap. Under the black dots, we see also other classes of dots.

Latent Dimension = 100: The combined test set yielded an MNIST accuracy of 
94.79% and an OOD accuracy of 88.67%, culmina=ng in a total accuracy of 91.73%. 
This is similar to the results obtained for a latent dimension of 10 and suggests a 
similar issue of under-representa=on of OOD instances in the higher-dimensional 
latent space.

Across all t-SNE visualiza=ons, we observe class overlap, which implies that there 
may s=ll be room for improving the model's capacity to delineate class boundaries in 
the latent space.

These observa=ons underscore the importance of carefully choosing the 
dimensionality of the latent space. A suitable balance ensures the model's 
robustness in handling in-distribu=on data and its proficiency in iden=fying out-ofdistribu=on instances. Excessively low or high dimensions can impede the model's 
effec=veness, hin=ng at an op=mal intermediate value, in this case, around 15.

## Best model results
1. Accuracies
  ![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/f72cf8a2-5356-4250-8c1a-962f43859aee)

2. TSNE Visualiza=on
     ![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/aa546937-8789-4dcc-b533-685af71d8638)

3. Confusion Matrix #1 - Baseline Model
   ![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/e5cc94e9-718e-4d1b-ad75-ed192037bb95)

   The confusion matrix visually shows our baseline model's performance. Each 
row and column signify actual and predicted classes, respec=vely. Diagonal 
elements are correctly predicted labels, while off-diagonal ones are 
misclassified. The baseline model effec=vely classified most MNIST dataset 
instances, with a majority of correct predic=ons along the diagonal. However, 
it misclassified some digits like 2s as 7s and 9s as 4s, likely due to their similar 
shapes.

4. Confusion Matrix #2 – OOD
   ![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/19c144e0-c3b5-4cec-bb62-95858d25f053)

   Analyzing the Out-of-Distribu=on (OOD) confusion matrix for binary 
classifica=on, we see high accuracy in dis=nguishing between 'Known' 
(MNIST) and 'Unknown' (OSR) classes. The model correctly classified 9892 
'Known' and 9911 'Unknown' instances. Misclassifica=ons were minimal, with 
only 108 'Known' instances classified as 'Unknown' and 89 'Unknown' 
instances classified as 'Known'. This performance underscores the model's 
robustness in handling unseen data, as indicated by the binary accuracy of 
99.015%.

5. Confusion Matrix #3 - OSR
   ![image](https://github.com/IdanArbiv/Open-Set-Recognition-With-Contrastive-Learning/assets/101040591/2e25bd17-4bbd-4e0d-9218-fc4b03f81afb)
   The provided confusion matrix illustrates our model's performance across ten 
MNIST classes and an 'Unknown' category represen=ng unseen data. High 
values along the diagonal indicate strong accuracy across all classes. 
Misclassifica=ons, represented by off-diagonal elements, are compara=vely 
few. The model demonstrates excellent iden=fica=on capabili=es for both 
MNIST and unseen data, as reflected in the high total accuracy and effec=ve 
handling of the 'Unknown' class.

6. Data Normaliza=on
   We use normaliza=on values 0f 0.5 for both mean and standard devia=on. 
While these may not correspond to the exact mean and std of the en=re data 
set, empirical results showed that using these normaliza=on values 
consistently yields the best performance across various datasets.

# Reference 
[1]  Toward Open Set Recognition
[2] arXiv - Towards Open Set Deep Networks
[3] arXiv - A Survey on Open Set Recognition
[4] arXiv - Rethinking InfoNCE: How Many Negative Samples Do You Need?
[5] arXiv - Representation Learning with Contrastive Predictive Coding
[6] arXiv - An Introduction to Variational Autoencoders











