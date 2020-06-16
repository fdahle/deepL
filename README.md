# deepL
Own deep learning framework written in Numpy. The number of layers and nodes is not given and can be changed by the user. 

 ## How to
```python
deepL = DeepLearner(type="classic") #init deepLearner
deepL.optimize(trainX, trainY) #train
testY = deepL.predict(testX) #evaluate
```

## Sources
based on following guides:
 - https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795 (Deep learning)
 - https://mlfromscratch.com/neural-network-tutorial/#/ (Deep learning)
 - https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1 (CNN)
 - https://victorzhou.com/blog/intro-to-cnns-part-1/ (CNN)
 - https://victorzhou.com/blog/intro-to-cnns-part-2/ (CNN)

## Roadmap
For next steps see the [Trello Board](https://trello.com/b/vFkWpaVW/deep-learning).
