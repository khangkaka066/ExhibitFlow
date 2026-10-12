# Floor geometry trial

This trial assumes the observed floor quadrilateral is 10 m wide (AB) and 12 m deep (AC). Values are approximate meters, not field measurements.

Point convention: A = near-left, B = near-right, C = far-left, D = far-right; homography input order is A-B-D-C.

| Case | Review | Camera/track/frame | Assumed distance | Reachable at 0.65 m? |
|---|---|---|---:|---|
| `user_reported_false_positive` | false_alarm_no_touch | 1/3/3868 | 0.76 m | no |
| `evt_000001` | pending | 1/11/7543 | 0.41 m | yes |

The known false positive and the remaining candidate are compared only to test whether floor geometry separates them. The 0.65 m reach threshold is diagnostic only. No operational threshold is activated until real dimensions are provided.
