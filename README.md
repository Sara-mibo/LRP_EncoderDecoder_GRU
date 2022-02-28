<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#about-the-project">Project Description</a></li>
    <li><a href="#getting-started">Dependencies</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Project Description

The reository provides the implementation of the Layerwise Relevance Propagation (LRP) method for GRU cells proposed by pytorch framework. We use LRP in order to explain the decisions of an encoder-decoder pollution forecasting model.   





<!-- GETTING STARTED -->

### Dependencies

The list of things you need to run the codes.
* environment lrpenv
  ```sh
  conda create -n lrpenv python=3.8
  ```
* activate lrpenv
  ```sh
  conda activate lrpenv
  ```
* pip
  ```sh
  conda install pip

  ```
* requirements
  ```sh
  pip install -r requirements.txt
  ```


<!-- USAGE EXAMPLES -->
## Usage

The folder LRP/ contains the main scripts of LRP implementation for a seq-2-seq model with GRU layers. 

The folder LRP_toyTask/ contains the scripts used for validation of the LRP implementation through a toy task.

The folder LRP_pollutionForecatModel/ contains the scripts used for applying LRP to a polluttion forecasting task.

The folder train_seq2seq_1hotEncoding/ contains the experiments which were done for the pollution forecasting task.


<p align="right">(<a href="#top">back to top</a>)</p>




<!-- LICENSE -->
## License

Distributed under the '' License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Sara Mirzavand Borujeni - sara.mirzavand.borujeni@hhi.fraunhofer.de - sarah.mb@outlook.com

Project Link: [https://github.com/Sara-mibo/LRP_EncoderDecoder_GRU](https://github.com/Sara-mibo/LRP_EncoderDecoder_GRU)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Design and Results of an AI-Based Forecasting of Air Pollutants for Smart Cities](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/VIII-4-W1-2021/89/2021/)
* [implementation of **Layer-wise Relevance Propagation (LRP) for LSTMs**, repo by L. Arras](https://raw.githubusercontent.com/ArrasL/LRP_for_LSTM)
* [Explaining Recurrent Neural Network Predictions in Sentiment Analysis, Arras, L., Montavon, G., Müller, K., & Samek, W.](https://aclanthology.org/W17-5221.pdf)

<p align="right">(<a href="#top">back to top</a>)</p>
