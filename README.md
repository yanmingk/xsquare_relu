# xsquare_relu
Some experiments about approximating x^2 with a ReLU network.

- `class_dataset.py`, `class_model.py`, `test_model.py`, and `train_model.py` define the classes and functions.
- `adam1.py` tries to optimize using the Adam optimizer and plots the maximal errors.
- `bfgs1.py` tries to optimize using the L-BFGS optimizerand plots the maximal errors.
- `bfgs-sgd.py` uses L-BFGS and SGD alternately.
- `gradients.py` records the reduction of gradients combined as one vector.
- `gradients2.py` records the reduction of gradients for each parameter seperately.

