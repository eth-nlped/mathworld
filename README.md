# mathworld

# World Models for Math Story Problems

This repository contains code for the paper:

### [World Models for Math Story Problems](https://arxiv.org/pdf/2306.04347.pdf) (Accepted at ACL Findings 2023).  
#### _Andreas Opedal, Niklas Stoehr, Abulhair Saparov and Mrinmaya Sachan_

Start by installing all packages:

`pip install -r requirements.txt`

## Citation
Please cite as:
```bibtex
@inproceedings{opedal-et-al-2023,
  author =      {Andreas Opedal and Niklas Stoehr and Abulhair Saparov and Mrinmaya Sachan},
  title =       {World Models for Math Story Problems},
  booktitle =   {Findings of the Association for Computational Linguistics: ACL 2023},
  year =        {2023},
  month =       jul,
  address = {Toronto, Canada},
  url = {https://arxiv.org/pdf/2306.04347.pdf},
}
```

## Demo

You will find a demo of our world model code in `mathworld_demo.ipynb`. It shows some of the basics, like how to 
* load an annotated world model and view its metadata
* visualize a world model
* get the logical forms of a world model
* apply the reasoning algorithm on a world model
* create a new world model from logical forms

## Data

The data with train and test splits can be found in `output_files/data`. You will find the annotated world model graphs stored in `.json` format in `output_files/data/{dataset}` (with dataset $\in$ [asdiv, mawps, svamp]). If you are only interested in the logical forms you have those available in `.csv` format in `output_files/data` (along with one sentence history for convenience).

## Code

All code associated with building and manipulating MathWorld world model objects is located in the `worldmodel` directory. The functions used to visualize world models is found in `utils/viz_helper.py`.

The rest of the code is specific to our paper. `preprocessing/` contains the code we used to preprocess the data from the various sources. `experiments/predictor` contains code used for the semantic parsing / solving (sec 5.1) and the generation (sec 5.3) experiments presented in the paper and `experiments/probing` contains the code used to run the knowledge probing experiments presented in the paper (sec 5.2).
