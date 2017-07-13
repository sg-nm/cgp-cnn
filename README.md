# Designing Convolutional Neural Network Architectures Based on Cartegian Genetic Programming

This repository contains the code for the following paper:

Masanori Suganuma, Shinichi Shirakawa, and Tomoharu Nagao, "A Genetic Programming Approach to Designing Convolutional Neural Network Architectures," 
Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17), pp. 497-504 (2017) [[DOI]](https://doi.org/10.1145/3071178.3071229) [[arXiv]](https://arxiv.org/abs/1704.00764)

## Requirement
We use the [Chainer](https://chainer.org/) framework for neural networks and tested on the following environment:

* Chainer version 1.16.0
* GPU: GTX 1080 or 1070
* Python version 3.5.2 (anaconda3-4.1.1)

## Usage

### Run the architecture search
This code can reproduce the experiment for CIFAR-10 dataset with the same setting of the GECCO 2017 paper (by default scenario). The (training) data are split into the training and validation data. The validation data are used for assigning the fitness to the generated architectures. We use the maximum validation accuracy in the last 10 epochs as the fitness value.

If you run with the ResSet described in the paper as the function set:

```shell
python exp_main.py -f ResSet
```

Or if you run with the ConvSet described in the paper:

```shell
python exp_main.py -f ConvSet
```

When you use the multiple GPUs, please specify the `-g` option:

```shell
python exp_main.py -f ConvSet -g 2
```

After the execution, the files, `network_info.pickle` and `log_cgp.txt` will be generated. The file `network_info.pickle` contains the information for Cartegian genetic programming (CGP) and `log_cgp.txt` contains the log of the optimization and discovered CNN architecture's genotype lists.

Some parameters (e.g., # rows and columns of CGP, and # epochs) can easily change by modifying the arguments in the script `exp_main.py`.

### Re-training

The discovered architecture is re-trained by the different training scheme (500 epoch training with momentum SGD) to polish up the network parameters. All training data are used for re-training, and the accuracy for the test data set is reported.

```shell
python exp_main.py -m retrain
```
