# model_comparisons
Code used to search for assess two modeling strategies - pairwise and metabolite mediated - for microbial community dynamics.

Paper abstract:

Personalized models of the gut microbiome are valuable for disease prevention and treatment. For this, one requires a mathematical model that predicts microbial community composition and the emergent behavior of microbial communities. Here, we compare two modeling strategies---pairwise modeling and metabolite mediated modeling. Our investigation into these two modeling frameworks revealed that metabolite mediated modeling was better able to capture emergent behavior in community composition dynamics than pairwise modeling.

Using publicly available data, we examine the ability of pairwise models and metabolite mediated models to predict trio growth experiments from the outcomes of pair growth experiments. We show that the underlying assumption of pairwise modeling does not fit growth data well. We specifically examine the Lotka-Volterra model and find it does not capture the emergent behavior of even small microbial communities. In contrast, metabolite mediated models can capture the emergent behavior of the community. Furthermore, metabolite mediated models can be used to explain a wide variety of interdependent growth data, and allow us to leverage advancements from the field of genome-scale metabolic models. We conclude that metabolite mediated modeling will be important in the development of accurate, clinically useful models of microbial communities.


This Repository:

This repository contains a number of scripts and functions used to assess modeling strategies.

estimate_parameters.py: Estimates model parameters from growth data.

load_gore_results.py: load the results from growth experiments published in Friedman et al

lv_pair_trio_functs.py: Contains most of the functions used, including to simulate deterministic and stochastic Lotka-Volterra, as well as to search for parameters.

met_mediated_fitting.py: build a metabolite mediated model that matches growth experiment outcomes.

non_add.py: Statistical arguments that community effects on growth rate are not pairwise additive.

search_for_parameters.py: Carry out "full experiment" and "pair experiment" to find LV parameters to fit the outcome data.

stochastic_model_analysis.py: Create realizations and run monte-carlo experiment on stochastic Lotka-Volterra.
