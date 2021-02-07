### Environment:
- conda create -n torch python=3.6
#install torchvision v0.2 on cuda v10.1 required by neural relational inference paper
- conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
- conda install matplotlib
- conda install networkx
#use higher framework for meta learning https://github.com/facebookresearch/higher
- pip install higher

#### run code on server
- screen
- ./run_script.sh 
- ./test.sh 

