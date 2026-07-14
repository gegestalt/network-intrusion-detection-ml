# NSL-KDD — Semi-Supervised Label-Budget Experiment

Only a stratified fraction of training labels is revealed. Self-training may help when pseudo-labels are reliable, but it can also amplify early mistakes.

| Method | Label fraction | Labelled rows | Accuracy | Macro-F1 | Normal recall | Attack recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| labelled_only_logreg | 0.01 | 1,259 | 0.7276 | 0.7261 | 0.9285 | 0.5755 |
| self_training_logreg | 0.01 | 1,259 | 0.7222 | 0.7205 | 0.9295 | 0.5653 |
| labelled_only_logreg | 0.05 | 6,299 | 0.7480 | 0.7474 | 0.9209 | 0.6171 |
| self_training_logreg | 0.05 | 6,299 | 0.7372 | 0.7364 | 0.9213 | 0.5979 |
| labelled_only_logreg | 0.10 | 12,597 | 0.7547 | 0.7542 | 0.9236 | 0.6268 |
| self_training_logreg | 0.10 | 12,597 | 0.7469 | 0.7462 | 0.9238 | 0.6130 |
| labelled_only_logreg | 0.25 | 31,494 | 0.7540 | 0.7535 | 0.9246 | 0.6249 |
| self_training_logreg | 0.25 | 31,494 | 0.7524 | 0.7519 | 0.9245 | 0.6221 |

## Interpretation

This track measures label efficiency. If self-training underperforms, that is a valid finding: pseudo-labels are not automatically trustworthy in shifted security data.
