| Method | workflow_completion_rate | modeled_completion_seconds | total_transfer_mb | recompute_seconds | handoff_failure_rate | reward |
|---|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 0.528 [0.278, 0.750] | 56.202 [40.574, 73.939] | 589.799 [174.324, 1087.161] | 20.001 [9.470, 31.113] | 0.319 [0.153, 0.528] | -10.299 [-20.895, -0.239] |
| PPO | 1.000 [1.000, 1.000] | 97.069 [67.527, 133.150] | 825.278 [240.491, 1526.933] | 60.035 [35.393, 87.809] | 1.000 [1.000, 1.000] | 6.517 [3.633, 9.705] |
| Controller-level MAPPO | 1.000 [1.000, 1.000] | 97.069 [67.301, 131.913] | 825.278 [227.689, 1514.824] | 60.035 [35.295, 87.743] | 1.000 [1.000, 1.000] | 6.517 [3.612, 9.687] |
| Two-step cost rule | 1.000 [1.000, 1.000] | 45.655 [34.329, 60.829] | 279.151 [57.473, 557.995] | 3.297 [1.319, 5.455] | 0.302 [0.094, 0.532] | 13.218 [11.262, 15.109] |

Values are evaluation-window means with percentile 95% bootstrap CIs (12 outer windows). 
Learned methods are averaged over seeds within each window before inference; the deterministic rule has one run per window. 
This is a calibrated non-formal pilot, not formal/holdout evidence or a real-RSU deployment result.
