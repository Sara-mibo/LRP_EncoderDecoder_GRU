<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#about-the-project">Project Description</a></li>
    <li><a href="#getting-started">Dependencies</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Project Description

This repository provides the implementation of the Layer-wise Relevance Propagation (LRP) explanation method for GRU cells (as proposed by the Pytorch framework), as well as for a sequence-to-sequence neural network architecture. We use LRP in order to explain the decisions of an encoder-decoder GRU-based pollution forecasting model.



<!-- GETTING STARTED -->
## Dependencies

The steps you need to run to install the required dependencies are the following:
* create environment lrpenv
  ```sh
  conda create -n lrpenv python=3.8
  ```
* activate environment lrpenv
  ```sh
  conda activate lrpenv
  ```
* install pip
  ```sh
  conda install pip

  ```
* install requirements
  ```sh
  pip install -r requirements.txt
  ```


<!-- USAGE EXAMPLES -->
## Usage

The folder `LRP/` contains the main part of the LRP implementation for a seq-2-seq model with GRU layers.

The folder `LRP_toyTask/` contains the scripts used for validation of the LRP implementation through a toy task.

The folder `LRP_pollutionForecastModel/` contains the scripts used for applying LRP to a pollution forecasting task.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Sara Mirzavand Borujeni: sara.mirzavand.borujeni@hhi.fraunhofer.de - sarah.mb@outlook.com

Project Link: [https://github.com/Sara-mibo/LRP_EncoderDecoder_GRU](https://github.com/Sara-mibo/LRP_EncoderDecoder_GRU)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Mirzavand Borujeni, S., Arras, L., Srinivasan, V. et al. Explainable sequence-to-sequence GRU neural network for pollution forecasting. Sci Rep 13, 9940 (2023).](https://doi.org/10.1038/s41598-023-35963-2)
* [Petry et al. 2021, Design and Results of an AI-Based Forecasting of Air Pollutants for Smart Cities, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., VIII-4/W1-2021, pages 89–96](https://doi.org/10.5194/isprs-annals-VIII-4-W1-2021-89-2021)
* [Reference Implementation of **Layer-wise Relevance Propagation (LRP) for LSTMs**, repository by L. Arras](https://github.com/ArrasL/LRP_for_LSTM)
* [Arras et al. 2017, Explaining Recurrent Neural Network Predictions in Sentiment Analysis, Proc. of the 8th Work. on Comput. Appr. to Subjectivity, Sentiment and Social Media Analysis, ACL, pages 159-168](https://aclanthology.org/W17-5221)
* [Bach et al. 2015, On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation, PLoS ONE 10(7): e0130140](https://doi.org/10.1371/journal.pone.0130140)

<p align="right">(<a href="#top">back to top</a>)</p>
