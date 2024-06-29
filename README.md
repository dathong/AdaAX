# AdaAX: Explaining Recurrent Neural Networks by Learning Automata with Adaptive States.
Welcome to the official repository of the paper  [AdaAX: Explaining Recurrent Neural Networks by Learning Automata with Adaptive States](https://dl.acm.org/doi/abs/10.1145/3534678.3539356) 

The code has been uploaded. You can run the code in the "synthetic" folder to generate the results for "11111" application. Updated: The code for Yelp Review application is uploaded. The running steps are still the same.

Please make sure you have Anaconda3 and Pytorch installed. We use Tensorflow 1.x for the experiments. Also, pythomata https://pypi.org/project/pythomata/ is needed for visualizing the DFA (installed by 'pip install pythomata'). Here are the instructions:
1) (Optional) Run the "utils.py" first to generate the dataset in "x.csv" and "y.csv" files. These files are uploaded too.
2) Run the "train_model.py" to train the model. It is advisable to wait until the accuracy is close to 1 (the higher accuracy, the better DFA can be generated)
3) Run the "load_model.py" to generate the temporary data files.
4) Run the "gen_rules_cluster_dfs.py" to generate the rules file ("inputs.txt")
5) Run the "dfa.py" to generate the DFA (will be in "vis.svg" file)

Please feel free to contact me at "hongtiendat@gmail.com" if you have any questions.

# Sample DFAs

<img src="/imgs/dfa1.png" width="500" height="150">
<img src="/imgs/dfa2.png" width="300" height="100">
<img src="/imgs/dfa3.png" width="400" height="800">

