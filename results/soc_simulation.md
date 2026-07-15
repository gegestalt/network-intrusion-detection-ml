# Operational SOC Simulation

Scenario: 1,000,000 daily flows, 0.50% malicious.

| Model | Threshold rule | Macro-F1 | Attack recall | False alerts/day | True alerts/day | Missed attacks/day | Total alerts/day | Operational precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LogReg | default_0.50 | 0.7535 | 0.6238 | 73567 | 3119 | 1881 | 76686 | 0.0407 |
| LogReg | validation_F1 | 0.7585 | 0.6361 | 78690 | 3180 | 1820 | 81871 | 0.0388 |
| LogReg | validation_F2_attack_recall_weighted | 0.7830 | 0.6905 | 94162 | 3452 | 1548 | 97614 | 0.0354 |
| LogReg_balanced | default_0.50 | 0.7539 | 0.6251 | 74489 | 3126 | 1874 | 77615 | 0.0403 |
| LogReg_balanced | validation_F1 | 0.7594 | 0.6383 | 79612 | 3191 | 1809 | 82804 | 0.0385 |
| LogReg_balanced | validation_F2_attack_recall_weighted | 0.7791 | 0.6813 | 91190 | 3406 | 1594 | 94597 | 0.0360 |
| ExtraTrees_balanced | default_0.50 | 0.7876 | 0.6487 | 27562 | 3244 | 1756 | 30806 | 0.1053 |
| ExtraTrees_balanced | validation_F1 | 0.7853 | 0.6447 | 27460 | 3224 | 1776 | 30683 | 0.1051 |
| ExtraTrees_balanced | validation_F2_attack_recall_weighted | 0.8126 | 0.7315 | 79612 | 3657 | 1343 | 83270 | 0.0439 |
| HistGB_balanced | default_0.50 | 0.7968 | 0.6650 | 28074 | 3325 | 1675 | 31399 | 0.1059 |
| HistGB_balanced | validation_F1 | 0.8159 | 0.6987 | 28894 | 3494 | 1506 | 32388 | 0.1079 |
| HistGB_balanced | validation_F2_attack_recall_weighted | 0.8230 | 0.7123 | 30431 | 3562 | 1438 | 33992 | 0.1048 |

## Interpretation

This table often changes which model looks best. A threshold that improves macro-F1 can still produce an impossible analyst workload if benign false alerts are too high.
