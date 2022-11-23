Introduction
============

The likelihood function is the fundamental ingredient of any statistical inference.
It encodes the full information on experimental measurements and allows for their interpretation from frequentist and Bayesian perspectives.

Current experimental and phenomenological results in fundamental physics and astrophysics typically involve complicated fits with several parameters of interest and hundreds of nuisance parameters.
Unfortunately, it is generically considered a hard task to provide all the information encoded in the likelihood function in a practical and reusable way.
Therefore, experimental analyses usually deliver only a small fraction of the full information contained in the likelihood function, typically in the form of confidence intervals obtained by profiling on the nuisance parameters, or in terms of probability intervals obtained by marginalising over nuisance parameters.

The DNNLikelihood project aims at encoding the full experimental likelihood function in a trained Deep Neural Network, which is able to reproduce the original likelihood function as a function of physical and nuisance parameters with the required accuracy.

.. include:: external_links.rst