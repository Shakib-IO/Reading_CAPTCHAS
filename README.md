# Reading_CAPTCHAS

# **Propose methods are CNNs, RNNs and CTC loss.**

So, for this implementation we are going to talk about only CTC (Connectionist Temporal Classification) loss.

> *Connectionist Temporal Classification (CTC) is a type of Neural Network output helpful in tackling sequence problems like handwriting and speech recognition where the timing varies. Using CTC ensures that one does not need an aligned dataset, which makes the training process more straightforward.*

---

<img src='https://miro.medium.com/max/1400/1*i2OG4hu9EjsyWcVMc4OOvA.png'>

---

CTC is simply a loss function that is used to train Neural Networks, like Cross-Entropy and so on. It is used at problems, where having aligned data is an issue, like Speech Recognition.

With CTC, there is no need for aligned data. That is because it can assign a probability for any label, given an input. This means it only requires an audio file as input and a corresponding transcription.  But how can CTC assign a probability for any label, just given an input? Like I said, CTC is "alignment-free“. It works by summing over the probability of all possible alignments between the input and the label.  

---
**How CTC works**

As already discussed, we don’t want to annotate the images at each horizontal position (which we call time-step from now on). The NN-training will be guided by the CTC loss function. We only feed the output matrix of the NN and the corresponding ground-truth (GT) text to the CTC loss function. But how does it know where each character occurs? Well, it does not know. Instead, it tries all possible alignments of the GT text in the image and takes the sum of all scores. This way, the score of a GT text is high if the sum over the alignment-scores has a high value.

---

CTC works on the following three major concepts:
- Encoding the text
- Loss calculation
- Decoding
***

**Encoding the Text**

The issue with methods not using CTC is, what to do when the character takes more than one time-step in the image? Non-CTC methods would fail here and give duplicate characters.

To solve this issue, CTC merges all the repeating characters into a single character. For example, if the word in the image is ‘hey’ where ‘h’ takes three time-steps, ‘e’ and ‘y’ take one time-step each. Then the output from the network using CTC will be ‘hhhey’, which as per our encoding scheme, gets collapsed to ‘hey’.

Now this question arises: What about the words where there are repeating characters? For handling those cases, CTC introduces a pseudo-character called blank denoted as “-“ in the following examples. While encoding the text, if a character repeats, then a blank is placed between the characters in the output text. Let’s consider the word ‘meet’, possible encodings for it will be, ‘mm-ee-ee-t’, ‘mmm-e-e-ttt’, wrong encoding will be ‘mm-eee-tt’, as it’ll result in ‘met’ when decoded. The CRNN is trained to output the encoded text.

---

**Loss calculation**

We need to calculate the loss value for the training samples (pairs of images and GT texts) to train the NN. You already know that the NN outputs a matrix containing a score for each character at each time-step. A minimalistic matrix is shown in Figure where there are two time-steps (t0, t1) and three characters (“a”, “b” and the blank “-”). The character-scores sum to 1 for each time-step.
<img src = 'https://miro.medium.com/max/800/1*BFQYgGofh6HOxnGdkJnO-w.png'>


Further, you already know that the loss is calculated by summing up all scores of all possible alignments of the GT text, this way it does not matter where the text appears in the image.

The score for one alignment (or path, as it is often called in the literature) is calculated by multiplying the corresponding character scores together. In the example shown above, the score for the path “aa” is 0.4·0.4=0.16 while it is 0.4·0.6=0.24 for “a-” and 0.6·0.4=0.24 for “-a”. To get the score for a given GT text, we sum over the scores of all paths corresponding to this text. Let’s assume the GT text is “a” in the example: we have to calculate all possible paths of length 2 (because the matrix has 2 time-steps), which are: “aa”, “a-” and “-a”. We already calculated the scores for these paths, so we just have to sum over them and get 0.4·0.4+0.4·0.6+0.6·0.4=0.64. If the GT text is assumed to be “--”, we see that there is only one corresponding path, namely “--”, which yields the overall score of 0.6·0.6=0.36.

We are now able to compute the probability of the GT text of a training sample, given the output matrix produced by the NN. The goal is to train the NN such that it outputs a high probability (ideally, a value of 1) for correct classifications. Therefore, we maximize the product of probabilities of correct classifications for the training dataset. For technical reasons, we re-formulate into an equivalent problem: minimize the loss of the training dataset, where the loss is the negative sum of log-probabilities. If you need the loss value for a single sample, simply compute the probability, take the logarithm, and put a minus in front of the result. To train the NN, the gradient of the loss with respect to the NN parameters (e.g., weights of convolutional kernels) is computed and used to update the parameters.

---
**Decoding**

Once CRNN gets trained, we want it to give us output on unseen text images. Putting it differently, we want the most likely text given an output matrix of the CRNN. One method can be to examine every potential text output, but it won’t be very practical to do from a computation point of view. The best path algorithm is used to overcome this issue.

It consists of the following two steps:
- Calculates the best path by considering the character with max probability at every time-step.
- This step involves removing blanks and duplicate characters, which results in the actual text.
For example, let us consider the matrix in Fig.3. If we consider the first time-step t0, then the character with maximum probability is ‘b’. For t1 and t2 character with maximum probability is ‘-‘ and ‘-‘ respectively. So, the output text according to the best path algorithm for matrix in Figure after decoding is ‘b’.

---

Resources:

- [Medium Blog](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c)

- [Siddhant's Scratch Book](https://sid2697.github.io/Blog_Sid/algorithm/2019/10/19/CTC-Loss.html?fbclid=IwAR2llk4Fb-yDLRESV4LluXan7zDsHNVt_rgWqxM6IhTMDZ5poVEDZwtzI2g)

- [Machine Learning -- Blog](https://machinelearning-blog.com/2018/09/05/753/)

---
