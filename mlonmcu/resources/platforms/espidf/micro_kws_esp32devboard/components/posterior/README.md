# MicroKWS Posterior Handler

This is an ESP-IDF component for the MicroKWS project implementing a posterior handler to improve the detection performance.

## Motivation

In order to derive a final decision of whether a keyword is present in an incoming audio stream one needs to further process the posteriors yielded by the ML model.

In the previous lab exercises, you trained your model in what is called *non-streaming mode*. You were doing a standard multi-class classification of independent input audio files, which each contained a single keyword. You trained your model on these 1s long audio vectors and observed the (hopefully) increasing accuracy with which your model was able to classify them. Your model operated on well-defined and isolated class realizations (i.e. the audio sample files). However, KWS, in general, is not a static task but a dynamic one.

For this reason, the input samples in the training process have been shifted randomly in the time domain and some additive background noise was added. For the deployment, we still need to adjust the code and the non-streaming model such that it can work in what is *streaming mode*. Dynamic streaming mode systems, in comparison to their non-streaming mode counterparts, do not only need to maintain an acceptable inference frequency, but also have to deal with inter-class transitions, overlapping speech samples, and partially spoken words.

Our system needs to be robust enough to deal with these real-world phenomena. An example algorithm that is going to be implemented in the Lab 2 Assignment is introduced in the following.

## Description
See section 2.2.1 in Lab 2 manual!

## Task

Implement the described behavior.

See comments and docstrings in `posterior.cc` and `include/posterior.h` for details!

## Unit Testing

To check your implemented solution a set of unit tests is provided in the `test` directory of this component. How to use them is explained in the `test/README.md` of the top-level MicroKWS project itself.
