# Neural Foundations: One Neuron and Activations

This is an educational foundation phase. It explains the machinery used by later MLP/CNN/RNN experiments before those models are treated as black boxes.

## Single Neuron Step

- weighted sum `z`: **0.420000**
- sigmoid prediction: **0.603483**
- binary cross-entropy loss: **0.505037**
- gradient wrt weights: `[-0.317213, 0.47582, -0.158607]`
- gradient wrt bias: **-0.396517**
- updated weights: `[0.331721, -0.247582, 0.115861]`
- updated bias: **-0.060348**

## Activation Interpretation

- **Sigmoid** is bounded and useful for binary output probabilities, but it saturates.
- **Tanh** is zero-centered but still saturates at large magnitude.
- **ReLU** is simple and efficient, but negative units can become inactive.
- **Leaky ReLU / ELU / GELU** keep some signal for negative inputs or smooth the transition.
- **Softmax** converts multiclass logits into class probabilities whose sum is one.

Activation table saved to `results/activation_functions.csv`.
Activation plot saved to `results/figures/neural_foundations_activation_functions.png`.
