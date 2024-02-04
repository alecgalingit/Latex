import json 
import os

SUBJECTS = {'Calculus':[
    "Limits and Continuity",
    "Differentiation",
    "Integration",
    "Differential Equations",
    "Sequences and Series",
    "Multivariable Calculus",
    "Partial Derivatives",
    "Multiple Integrals",
    "Vector Calculus",
    "Line Integrals",
    "Surface Integrals",
    "Green's Theorem",
    "Divergence Theorem",
    "Stokes' Theorem"
],'Linear Algebra':["Systems of Linear Equations",
    "Matrices and Matrix Algebra",
    "Determinants",
    "Vector Spaces",
    "Linear Transformations",
    "Eigenvalues and Eigenvectors",
    "Orthogonality",
    "Inner Product Spaces",
    "Diagonalization",
    "Singular Value Decomposition",
    "Norms and Metrics"],'differential equations':["First-Order Differential Equations",
    "Second-Order Linear Differential Equations",
    "Higher-Order Linear Differential Equations",
    "Systems of Differential Equations",
    "Series Solutions of Differential Equations",
    "Transform Methods (Laplace Transform, Fourier Transform)",
    "Partial Differential Equations",
    "Boundary Value Problems",
    "Stability and Bifurcation"],'Statistics':["Descriptive Statistics",
    "Probability Distributions",
    "Statistical Inference",
    "Hypothesis Testing",
    "Confidence Intervals",
    "Regression Analysis",
    "Analysis of Variance (ANOVA)",
    "Non-parametric Statistics",
    "Bayesian Statistics",
    "Experimental Design",
    "Multivariate Statistics",
    "Time Series Analysis"]}

BASEDIR = '/Users/alecsmac/coding/latex/'

with open(os.path.join(BASEDIR,'misc','subjects.json'), 'w') as json_file:
    json.dump(SUBJECTS, json_file)