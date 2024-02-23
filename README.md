# RVE
This repo reveals the code for paper "Simulation-Free Determination of Microstructure Representative Volume Element Size via Fisher Scores".

Data use for experiments in the RVE paper:

"experi1.mat": .matlab file for the 2000x2000 ($2\mu m\times 2\mu m$) micrograph used in experiment 1;

"experi2_15.mat": .matlab file for the 2000x2000 ($15\mu m\times 15\mu m$) micrograph used in experiment 2;

"experi2_30.mat": .matlab file for the 2000x2000 ($30\mu m\times 30\mu m)$ micrograph used in experiment 2;

"experi3.mat": .matlab file for the 4000x4000 ($60\mu m\times 60\mu m)$ micrograph used in experiment 3.

To run the algorithm, download the .mat files in data folder, revise the path in read_data.py, and run the below command.
```
python3 main.py --data='2um' --model='nnet' --filename='2um_nnet'
```
