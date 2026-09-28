# AUT-2802
Codes for the Presentation as a part of the course AUT-2802
We train LeNet-5 on MNIST with different optimizers (SGD with momentum, and Adam) and different learning rates, deliberately without any stabilising training techniques (no batch norm, no learning-rate schedules, no gradient clipping). The goal is to see the raw behaviour of each optimizer, including how and when it fails.


The model -
LeNet-5 is a small convolutional neural network with about 61,700 trainable parameters.
Input 1×32×32

  → Conv 5×5, 6 filters   + activation   → 6×28×28
  → Pool 2×2                             → 6×14×14
  → Conv 5×5, 16 filters  + activation   → 16×10×10
  → Pool 2×2                             → 16×5×5
  → Conv 5×5, 120 filters + activation   → 120×1×1
  → Flatten                              → 120
  → Linear 120→84         + activation   → 84
  → Linear 84→10                         → 10 class scores
MNIST images are 28×28, so they are padded to 32×32 (with the normalised background value) to match what LeNet expects. Note that this is the common simplified LeNet-5: the convolution in the third layer connects every input map to every filter, rather than using the sparse connection table from the original paper.

The experiment - 

Data split. 54,000 training images, 6,000 validation images (a fixed split, identical for every run) and the standard 10,000 test images. Pixels are normalised with mean 0.1307 and standard deviation 0.3081.

Training setup. 15 epochs, batch size 128, cross-entropy loss, PyTorch defaults for weight initialisation. No batch normalisation, no learning-rate schedule, no gradient clipping, no data augmentation. Seeds. Every configuration is run with 3 seeds (37, 42, 73). The seed controls both the initial weights and the order of the training batches. Since no stabilising techniques are used, results can vary a lot from run to run, so a single run can be misleading. Running three seeds shows which results are reliable.

Total: 15 configurations × 3 seeds × 2 architectures = 90 training runs.
Group	Settings
SGD	Full 3×3 grid: learning rate ∈ {0.001, 0.01, 0.1} × momentum ∈ {0, 0.5, 0.9}
Adam	Learning rate ∈ {1e-4, 1e-3, 3e-3, 1e-2, 3e-2}
Stress test	SGD with lr = 0.1 and momentum = 0.99 (momentum deliberately pushed too far)


Results -

Each notebook produces three CSV files. The history file holds the training and validation loss and accuracy for every epoch of every run. The test file holds the final validation and test metrics for every run, along with whether the run diverged. The summary file holds the mean and standard deviation across seeds for each configuration, together with a verdict: strong if test accuracy is above 98%, works if above 90%, and underfits otherwise.

Each architecture also has six figures in the figures folder, prefixed with classic_ or Relu_. The first shows training loss on a log scale for each optimizer and learning rate. The second shows validation accuracy over the epochs. The third shows the effect of momentum at a fixed learning rate. The fourth is a heatmap of SGD validation accuracy across learning rate and momentum. The fifth plots accuracy against learning rate for SGD and Adam. The sixth is a bar chart of final test accuracy for every configuration. In the line plots, the solid line is the mean over the three seeds and the shaded band is the min to max range, so a wide band means an unreliable configuration.

The best result for the tanh and average pooling network was SGD with learning rate 0.1 and momentum 0.5, at 99.03% test accuracy. The best for the ReLU and max pooling network was Adam with learning rate 0.001, at 98.99%. Several other configurations land within about 0.1% of these, which is inside seed-to-seed noise, so the full table in the summary files should not be read as a strict ranking.

Key findings - 

There is no single best optimizer, only a best pairing of optimizer and learning rate. The spread within SGD across learning rates, from about 79% to 99% for the tanh network, is far larger than the gap between SGD and Adam at their best settings.

Momentum acts like a learning rate multiplier, with the effective step being roughly the learning rate divided by one minus the momentum. At a small learning rate of 0.001 it rescues training, taking the tanh network from 79% with no momentum to 96% with momentum 0.9. At a large learning rate combined with too much momentum (0.1 and 0.99), training collapses, to about 21% for the tanh network and to chance level of roughly 10% for the ReLU network.

Adam is not tuning free. It has its own usable window, with 0.001 working best here and 0.03 being too high. At 0.03 the tanh network reached only 88.8%, and for the ReLU network one seed trained to 93.5% while the other two collapsed to chance level at 11.4%. Adam's main practical advantage in these runs was faster progress in the first few epochs.

Using several seeds matters. A single run of the Adam 0.03 ReLU configuration could have looked fine or completely broken depending on the seed.

The ReLU and max pooling network was more forgiving at small learning rates. For SGD with learning rate 0.001 and momentum 0.9 it reached 98.3% against 95.9% for the tanh network. The tanh network, however, handled the largest SGD step with momentum 0.9 better, scoring 98.98% against 97.6%.

Conclusion - 

This section answers the three points of the assigned task directly.

Task 1: Show and discuss optimizers in LeNet-5 on MNIST without stable training techniques. We trained LeNet-5 on MNIST using a plain training loop with no batch normalisation, no learning rate schedule, no gradient clipping and no data augmentation, so that the behaviour seen comes from the optimizer and its settings alone. An optimizer decides how far and in what manner the weights are updated after each backward pass, and everything else in training (forward pass, loss, gradients) is identical across our runs. To check that the conclusions do not depend on one particular network, we repeated the whole study on two versions of LeNet-5: the original tanh with average pooling, and the modern ReLU with max pooling. Without stabilising techniques the optimizer has nothing to fall back on, which is why poorly chosen settings visibly fail here, either by learning extremely slowly or by collapsing to chance level accuracy of about 10%.

Task 2: Try SGD and Adam with different learning rates and momentum. For SGD we ran a full grid of learning rates 0.001, 0.01 and 0.1 against momentum values 0, 0.5 and 0.9, plus one deliberately extreme run with learning rate 0.1 and momentum 0.99. For Adam we ran five learning rates from 0.0001 to 0.03. Every configuration was repeated with three seeds (37, 42 and 73), giving 45 runs per architecture and 90 in total. The main outcomes were as follows. Learning rate is the dominant factor: for SGD without momentum on the tanh network, moving the learning rate from 0.001 to 0.1 took test accuracy from 79.4% to 99.0%. Momentum behaves like a multiplier on the learning rate, with an effective step of roughly the learning rate divided by one minus the momentum. It helps when the step is too small (at learning rate 0.001, momentum 0.9 lifted the tanh network from 79.4% to 95.9%) and hurts when the step is already large (learning rate 0.1 with momentum 0.99 collapsed to 21.2% on the tanh network and 9.9% on the ReLU network). Adam performed best at its learning rate of 0.001 (98.88% on tanh, 98.99% on ReLU), degraded at 0.0001 because the steps were too small, and became unreliable at 0.03, where the ReLU network trained on one seed (93.5%) but collapsed to chance level on the other two (11.4%).

Task 3: Show and compare the training plots for the different optimizers and values used, and the test results. The training and validation plots in the figures folder (loss curves, validation accuracy curves, the momentum comparison, the SGD heatmap and the learning rate sensitivity plot) show the same story from different angles, and the final bar chart and summary files give the test results. At their best settings the optimizers are essentially tied: the best tanh result was SGD with learning rate 0.1 and momentum 0.5 at 99.03% test accuracy, and the best ReLU result was Adam with learning rate 0.001 at 98.99%, with several other configurations within about 0.1%. That gap is inside seed-to-seed noise, so it should not be used to declare a winner. What the plots do show clearly is that the spread caused by learning rate and momentum choices (roughly 10% to 99%) is far larger than the difference between SGD and Adam when each is well tuned. Adam reached about 96% validation accuracy after one epoch at learning rate 0.001, but tuned SGD matched that speed, so Adam's practical benefit is that a sensible default works out of the box, not that it is fundamentally better.

Overall conclusion. Without stabilising techniques, the choice of optimizer matters less than the choice of its hyperparameters. A well-chosen learning rate (and momentum, for SGD) gives 98% to 99% test accuracy with either optimizer, while a poorly chosen one can leave the network barely trained or completely collapsed. Momentum is best understood as a way of scaling the learning rate rather than a free improvement. Adam is more forgiving to set up, since its useful range sits around 0.001, but it is not tuning free and fails at high learning rates. Running several seeds was essential, because some configurations gave very different outcomes from one seed to the next. Finally, because MNIST is easy and LeNet-5 is small, these results show the pattern of how optimizers succeed and fail rather than a definitive ranking, and stabilising techniques such as batch normalisation, learning rate schedules or gradient clipping would be the natural next step to make training less sensitive to these choices.

