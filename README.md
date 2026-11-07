# A Distribution Mapping Approach to Counterfactually Fair Reinforcement Learning

This repository contains the implementation of the paper "A Distribution Mapping Approach to Counterfactually Fair Reinforcement Learning" (NeurIPS 2026).

## Abstract of the paper

Reinforcement learning (RL) seeks to optimize sequential decisions to maximize population-level benefits over time. However, when deployed in high-stakes settings such as healthcare, RL decisions might systematically restrict some subpopulation's access to valuable resources or services in a manner contrary to the values and goals of stakeholders. Counterfactual fairness (CF) offers a promising framework to address this problem based on causal reasoning. This paper develops a data preprocessing algorithm that, when used in tandem with policy learning, enables CF in RL. Our algorithm relies on a novel quantile distribution mapping method for sequentially estimating the counterfactual states and rewards in the data preprocessing step, subsuming common additivity assumptions used for counterfactual prediction as special case. We theoretically prove that the per-step level of counterfactual unfairness and infinite-horizon suboptimality gap can be bounded under mild regularity conditions. We also empirically test our algorithm in numerical experiments as well as in application to a real-world interventional digital health dataset.

## Requirements

All experiments were conducted using the Anaconda environment provided by the computing cluster, loaded using:

```setup
module load python3.10-anaconda/2023.03
```

The Python packages required for the experiments are listed in ``requirements.txt``.

## Instructions for Reproducing the Numerical Experiments

### Reproducing Main Experiments

To run the experiment presented in Section 5 of the main paper, go to the directory ``cfsmdm_code/numerical_experiment`` and run this command:

```
python simulation_m1_N.py
python simulation_m1_Z.py
```

To produce the corresponding figure, run ``cfsmdm_code/numerical_experiment/plot_m1.R`` after running the above command.

To run the experiment presented in Section L1 of the appendix, go to the directory ``cfsmdm_code/numerical_experiment`` and run this command:

```
python simulation_m2_N.py
python simulation_m2_Z.py
```

To produce the corresponding figure, run ``cfsmdm_code/numerical_experiment/plot_m2.R`` after running the above command.

### Reproducing Additional Experiment on CF Metric and $T$

To run the experiment presented in Section L2 of the appendix, go to the directory ``cfsmdm_code/aux_numerical_experiment`` and run this command:

```
python simulation_m1_T.py
python simulation_m2_T.py
```

The above command runs the experiment for all baselines except FLAP_M and ECOCF_M. To run experiment for FLAP_M and ECOCF_M, go to the directory ``cfsmdm_code/numerical_experiment`` and run this command:

```
python simulation_m1_T_add.py
python simulation_m2_T_add.py
```

After running these commands, add contents from ``numerical_experiment/results/results_m1_T.csv`` to ``aux_numerical_experiment/results_aux/results_m1_T.csv``, and add contents from ``numerical_experiment/results/results_m2_T.csv`` to ``aux_numerical_experiment/results_aux/results_m2_T.csv``.

To produce the corresponding figure, run ``cfsmdm_code/aux_numerical_experiment/plot_T.R`` after the steps above.

### Reproducing Additional Experiment on Higher-dimensional CMDP (CMDP 3)

To run the experiment presented in Section L3 of the appendix, go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m3_N.py
```

To produce the corresponding figure, run ``cfsmdm_code/additional_numerical_experiment/plot_m3.R`` after the steps above.

### Reproducing Additional Experiment on the Performance of CFSMDM Based on Demographic Parity

To run the experiment presented in Section L4 of the appendix, go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m1_N_dp.py
python simulation_m2_N_dp.py
```

To produce the corresponding figure, run ``cfsmdm_code/additional_numerical_experiment/plot_dp.R`` after the steps above.

### Reproducing Additional Experiment on Sensitivity to Violations of the Single Scalar Noise Formulation

To run the experiment presented in Section L5 of the appendix, go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m4_single_scalar_noise.py
python simulation_m5_single_scalar_noise.py
```

To produce the corresponding figure, run ``cfsmdm_code/additional_numerical_experiment/plot_single_scalar_noise.R`` after the steps above.

### Reproducing Additional Experiment on Sensitivity to Violations of the Strict Monotonicity Assumption

To run the experiment presented in Section L6 of the appendix, go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m6_strict_monotonicity.py
python simulation_m7_strict_monotonicity.py
```

To produce the corresponding figure, run ``cfsmdm_code/additional_numerical_experiment/plot_strict_monotonicity.R`` after the steps above.

### Reproducing Additional Experiments Using CQL for Policy Learning

To run the experiment presented in Section L7 of the appendix, first go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m1_N_cql.py
python simulation_m1_Z_cql.py
```

This runs the experiments using CMDP 1. To produce the corresponding figure, run ``cfsmdm_code/additional_numerical_experiment/plot_m1_cql.R`` after running the above command.

Then, run this command:

```
python simulation_m2_N_cql.py
python simulation_m2_Z_cql.py
```

This runs the experiments using CMDP 2. To produce the corresponding figure, run ``cfsmdm_code/additional_numerical_experiment/plot_m2_cql.R`` after running the above command.

### Reproducing Additional Experiment on Sensitivity of CFSMDM to the Choice of Quantile Estimator

To run the experiment presented in Section L8 of the appendix, go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m1_qr_ablation.py
python simulation_m2_qr_ablation.py
```

### Reproducing Additional Experiment on Sensitivity of CFSMDM to the Number of Quantiles in the Quantile Grid

To run the experiment presented in Section L9 of the appendix, go to the directory ``cfsmdm_code/aux_numerical_experiment`` and run this command:

```
python simulation_m1_q.py
python simulation_m2_q.py
```

### Reproducing Additional Experiment on the Computing Time of CFSMDM

To run the experiment presented in Section L9 of the appendix, go to the directory ``cfsmdm_code/additional_numerical_experiment`` and run this command:

```
python simulation_m1_profiling.py
```

## Instructions for Reproducing the Real Data Analysis

To run the PowerED real data analysis in Section 6 of the main paper, go to the directory ``cfsmdm_code/real_data_analysis_powered`` and run this command:

```
python run_rda_sex.py
python run_rda_age.py
python run_rda_ethnicity.py
python run_rda_education.py
```

To run the Intern Health Study real data analysis in Section M of the appendix, go to the directory ``cfsmdm_code/real_data_analysis_ihs`` and run this command:

```
python run_rda_mood_full_data.py
```

Please note that the real data analysis code in these folders is not executable because the PowerED and Intern Health Study datasets used in the analysis are not included in the folder. However, interested readers can request access to the datasets following the instructions in the "Data and Code Availability Statement" section of the paper.

## Instructions for Reproducing the Semi-synthetic Experiments

To run the PowerED semi-synthetic experiments in Section N of the appendix, go to the directory ``cfsmdm_code/semi_synthetic_experiment_powered`` and run this command:

```
python simulation_sex.py
python simulation_age.py
python simulation_ethnicity.py
python simulation_education.py
```

To run the Intern Health Study semi-synthetic experiment in Section N of the appendix, go to the directory ``cfsmdm_code/semi_synthetic_experiment_ihs`` and run this command:

```
python simulation_sex.py
```

Please note that the semi-synthetic experiment code in this folder is not executable because the PowerED and Intern Health Study datasets used in the experiment is not included in the folder. However, interested readers can request access to the datasets following the instructions in the "Data and Code Availability Statement" section of the paper.

## Results

Performance of the proposed method is summarized in Sections 5 and 6 of the main paper and Sections L, M, and N of the appendix. Please note that the results might not always be exactly the same due to different computing environments and orders of code execution, but the main takeaway from the results is expected to be similar.

## License

This code adopts the MIT license.