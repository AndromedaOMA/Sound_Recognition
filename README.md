<h1 align="center">Hi 👋, here we have my Sound Recognition project</h1>
<h3 align="center">Developed and trained a Convolutional Neural Network for Sound Recognition task!</h3>


## Table Of Content
* [About Project](#project)
* [Architecture Overview](#architecture)
* [Dataset](#dataset)
* [Getting Started](#getting-started)

--------------------------------------------------------------------------------
<h1 id="project" align="left">🤖 About Project</h1>

The main objective of this project is to recognize speech and provide transcripts associated with speech in real time. The architecture is compact but efficient, leveraging a well-structured dataset provided by the **TorchAudio** library.

---

<h1 id="architecture" align="left">🧠 Architecture Overview</h1>

The architecture of the **Sound Recognition** model is quite simple, but effective. The SoundRecognitionModel is a convolutional neural network designed for classifying audio signals based on their spectrogram representations. The model takes as input a tensor of shape (1, 64, 44), which typically corresponds to a mono audio signal processed into a 2D spectrogram with 64 frequency bins and 44 time steps.

The architecture consists of two main parts: a **convolutional block** and a **fully connected block**.

The **convolutional block** contains three convolutional layers. Each layer is followed by batch normalization, a GELU activation function, and a max pooling operation. The first convolutional layer transforms the input from one channel to a configurable number of intermediate channels, defined by the parameter mid_channels. The next two convolutional layers maintain the same number of channels and continue to refine the feature maps. After each convolutional step, max pooling reduces the spatial dimensions, which helps the model capture higher-level features while decreasing the computational load.

After passing through the convolutional layers, the resulting tensor typically has the shape (mid_channels, 2, 4), which is then flattened into a one-dimensional vector of length 8 times the number of intermediate channels. This flattened vector is passed through the **fully connected block**. The first linear layer reduces the dimension to mid_channels, applies a GELU activation, and includes a dropout layer with a probability of 0.1 to prevent overfitting. The second linear layer maps the features to a 10-dimensional vector, corresponding to the number of target classes. A softmax function is applied at the end to produce a probability distribution over these 10 classes.

The model is **highly configurable**, allowing adjustments to the number of channels, kernel sizes, strides, paddings, and pooling parameters through an external ModelConfigs configuration class. It uses GELU for smooth nonlinear activation, batch normalization for faster convergence, and dropout for regularization. Overall, the SoundRecognitionModel is a compact and efficient network tailored for sound classification tasks, suitable for both training and deployment in real-time systems.

---

<h1 id="dataset" align="left">📄 Dataset</h1>

Dataset provided via the **TorchAudio** library. The labeled dataset is **UrbanSound8K**: where the input is given by audio sequences with a sampling rate of 8k, and the target is given by urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music associated with the audio sequences.

Next, we will pad each sample in the set to a standard size. Before loading and injecting the dataset into the model, the dataset was transformed into **Mel Spectrograms** so that the model, which contains convolutional layers, could process them efficiently.

---

<h1 id="getting-started" align="left">🚀 Getting Started</h1>

1. Clone the repository:
``` git clone git@github.com:AndromedaOMA/Sound_Recognition.git ```
2. Have fun!

---

> 📝 **Note**:  
> By completing this project, we emphasized our knowledge of Deep Learning in order to develop and complete the bachelor's project that solves the Speech Enhancement task.

* [Table Of Content](#table-of-content)
