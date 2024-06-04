# Fair Link Prediction with Multi-Armed Bandit Algorithms
This repository provides a reference implementation of **FairLink** framework as described in [1].

## How to use:

The file ```main.py``` contains the example code to use FairLink. FairLink expects as input the protected feature value(community membership), training graph, and negative/positive link examples. The input files for Bowdoin College facebook friendship network and Pokec friendship network in two different regions can be found in ```./data/graph/``` directory in pickle binary format. 


To run the algorithm on Pokec friendship network in Žilinský kraj Bytča, execute the following command from the project home directory:
```shell-script
python main.py
```
you can set the baseline algorithm using --algorithm option. For example to use preferential attachment:

```shell-script
python main.py --algorithm prf
```
you can change the input network to Bowdoin College facebook friendship network as follow:

```shell-script
python main.py --file-name bowdoin.pk
```
you can check other available options by following command:

```shell-script
python main.py --help
usage: main.py [-h] [--folder-path [FOLDER_PATH]] [--file-name [FILE_NAME]]
               [--algorithm [ALGORITHM]] [--test-size TEST_SIZE]
               [--add-radio ADD_RADIO] [--batch-step BATCH_STEP]
               [--acc-bound ACC_BOUND] [--gamma GAMMA]
               [--slot-number SLOT_NUMBER] [--epsilon EPSILON] [--decay DECAY]
               [--cross-validate CROSS_VALIDATE] [--file FILE]

Run FairLink.

optional arguments:
  -h, --help            show this help message and exit
  --folder-path [FOLDER_PATH]
                        path to input graph directory
  --file-name [FILE_NAME]
                        data file name
  --algorithm [ALGORITHM]
                        name of link prediciton baseline algorithm to use: jac
                        for Jacard, adar for adamic_adar and prf for
                        preferential_attachment. Default is jac
  --test-size TEST_SIZE
                        link prediction test size. Default is 0.8
  --add-radio ADD_RADIO
                        precent of test to be predicted .Default is 0.1
  --batch-step BATCH_STEP
                        batch step size. Default is 10
  --acc-bound ACC_BOUND
                        The bottom line of tolerance for accuracy during
                        GridSearch. Default is 0.5
  --gamma GAMMA         importance weight between accuracy and benefit during
                        Reward Update. Default is 1.0
  --slot-number SLOT_NUMBER
                        number of generated slots. Default is 20
  --epsilon EPSILON     Epsilon-Greedy algorithm hyper-parameter. Default is
                        0.3
  --decay DECAY         decay parameter during benefit update. Default is 0.8
  --cross-validate CROSS_VALIDATE
                        cross validate hyper-parameter in grid search. Default
                        is 3
  --file FILE
```

The Python libraries version information is provided in ```requirements.txt``` file.
```shell-script
pip install -r requirements.txt
```
### Refrence
[1] **Fair Link Prediction with Multi-Armed Bandit Algorithms** Wang W, Soundarajan S, in Association for the 15th ACM Web Science Conference (WebSci), 2023.
