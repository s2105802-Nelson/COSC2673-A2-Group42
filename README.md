<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<div align="center">
<h3 align="center">Image Classification of Cells for Cancerous Cell Type</h3>

  <p align="center">
    With PyTorch and Tensorflow based Neural Networks and Convolutional Neural Networks (CNNs)
    <br />
  </p>
</div>


<em>ReadMe is a Work in Progress</em>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>    
    <li><a href="#baseline-models">Baseline Models</a></li>
    <li><a href="#final-models">Final Models</a></li>
    <li><a href="#requirements">Requirements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<p>
The aim of this project is to train two machine learning models for image classification of the provided images of cells to predict whether the cells are cancerous and what type of cell they are. Inputs to these models is a dataset of over 20,000 images, that 
are 27x27 pixel images of cells from histopathological images. Around half of the images are labelled, the rest are unlabelled. The first model will be referred to as the "Cancerous Binary" model, which aims to predict a Binary target label, which is simply whether the cell is cancerous or not.
The second model will be referred to the "Cell Type Multi-class" model, which aims to predict whether the cell is of one of four types, epithelial, fibroblast, inflammatory or other.
</p>

<p>
Initially, Baseline models were trained and evaluated. Then further experimentation with different algorithms, modelling improvements and hyperparameter tuning were undertaken until final, best performing models were found. 
This project was directed to train models from scratch (i.e. no Transfer Learning), where processes should be able to run on a free Google Colab account (so without significant GPU resources required)
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Baseline Models

<p><strong>File: 05c.PyTorchBaseline.ipynb</strong></p>
<p>
For both models, a simple Fully Connected Neural Network model is used for baseline models, with no image preprocessing and the basic SGD optimiser. 
The Cancerous Binary baseline model, has predictions with a Training F1 of 0.828 and a Test F1 of 0.891. The F1 performance is good but could be improved upon, with a possible indication of bias in the model. The F1 error gap between the Training accuracy 
and the Test accuracy is low, meaning this model is generalizing well.
For the Cell Type Multi-class model, the M-Baseline predictions had a Training F1 of 0.668 and a Test F1 of 0.764. The F1 performance is low, indicating a strong possibility of bias. Additionally, the model is predicting significantly better on unseen test data. This is an indication that the baseline model is performing poorly.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Final Models

<p><strong>File: 28.PyTorchCNN07.ipynb</strong></p>
<p>
The Cancerous Binary final model, the recommended model for this task is a simple Convolutional Neural Network (CNN) with a Training F1 of 0.912 and a Test F1 of 0.899. This model has almost as good F1 and has a low amount of variance. 
The reason for this selection is that the model is much simpler in the convolution and classification layers. While other, more complex CNN models would take multiple hours to train, the B-11 trains much faster, with comparable performance.
With the Multi-class Cell Type Modelling task, the recommended model is as similarly structured simple CNN from the same file, with a Training F1 of 0.833 and a Test F1 of 0.797. Similarly, this model is efficient in training time and performs well.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Requirements

<em>Work in Progress</em>

<p>
This section and the instructions are currently a work-in-progress, to be updated with more specific instructions. However, here are some rough requirements on requirements to run notebooks for the project.
</p>

<ul>
  <li>A Python 3.7 or later environment</li>
  <li>Python Jupyter Notebooks. One recommended environment to run Jupyter Notebooks is from within Visual Studio Code, with the Jupyter Notebooks extension.</li>
  <li>The following Python Libraries, which can be installed using pip:
    <ul>
      <li>Pandas</li>
      <li>Numpy</li>
      <li>sklearn</li>
      <li>tensorflow</li>
      <li>pytorch</li>
      <li>torchvision</li>
      <li>matplotlib</li>
      <li>ggplot2</li>
    </ul>
  </li>
</ul>
