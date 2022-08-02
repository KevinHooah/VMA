# VMA
Domain-Variance-and-Modality-Aware-Model-Transfer

This repo is for the paper: [VMA: Domain Variance- and Modality-Aware Model Transfer for Fine-Grained Occupant Activity Recognition](https://ieeexplore.ieee.org/abstract/document/9826076) at IPSN 2022.

To run the demo code, please download the data from this [link](https://drive.google.com/file/d/1j2M1KRbOmdycxWckAnOpyh3I44vizkOw/view?usp=sharing).
Please unzip it first to the project directory and open the `run_demo_paper.ipynb`. Please run the notebook follows the guidlines inside.

The dataset of the project can be found at [Zenodo](https://zenodo.org/). Please follow the data pre-process procedure described in the paper if you want to extract the hand-crafted feature. For end-to-end deep feature (which is what I am highly interested in and encourage everyone to try), their is a toy model in the `demo_future_direction.ipynb` for you to play with.

If you find this code or dataset is usefull, we will be glad if you can cite us in your paper :-)

Recommended Packages:
* Python                    3.8+
* Numpy                     1.19.5
* Scikit-learn              1.0.1
* Pandas                    1.4.0
* Tensorflow                2.8.0

If you are using an Intel chip, you may need this to accelerate the computing:
* scikit-learn-intelex      2021.2.2
