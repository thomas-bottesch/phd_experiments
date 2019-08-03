# PhD Experiments
This repository provides the means to easily reproduce the experimental results of my PhD thesis.

To reproduce the results a Linux machine is needed. The following procedure was only tested on Ubuntu 18.04
but should work on other Debian Linux machines.

The experiments can be reproduced by executing just one command. Executing the command will do the following:
* Download and install all software required to run the tests (This requires installing via apt get using sudo)
* Create the environment to run the test
* Install specific versions of all required dependencies (numpy, sklearn, fcl)
* Automatically download the datasets that were used in the PhD experiments
* Execute the experiment
* Create the results as latex files

# Steps to run the experiments

1. Checkout this repo into a a folder <repo_folder>
2. The experiments can be started with:
 * python3 <repo_folder>start_experiment kmeans            (The experiments run for multiple month)
   Results will be available in the following files:
   * <repo_folder>/kmeans/output_path_latex/kmeans/*-single.tex
 * python3 <repo_folder>start_experiment kmeanspp          (The experiments run for multiple month)
   * <repo_folder>/kmeanspp/output_path_latex/kmeanspp/*-single.tex
 * python3 <repo_folder>start_experiment minibatch_kmeans  (The experiments run for multiple month)
   * <repo_folder>/minibatch_kmeans/output_path_latex/minibatch_kmeans/*-single.tex

# Results

The following results are creates as .tex files.

## plot-param-search-single.tex

Contains latex plots for every tested dataset and for all tested feature-maps showing how the speed-up changes
when varying their parameter.

## plot-iteration-duration-single.tex

For every experiment and datasets the duration for every single iteration is shown in a plot. These
plots are configurable (in the latex document) by specifying a primary and secondary algorithm. 
* \def\primaryalg{<algorithm_name>}
* \def\secondaryalg{<another_algorithm_name>}

The algorithmnames that can be used are specified within the .tex file.

## tbl-speed-comparison-*-single.tex

Creates a latex table which compares algorithms and their speed-ups on the datasets.

## tbl-memory-comparison-*-single.tex

Creates a latex table which compares the algorithms memory consumption while clustering the different datasets.