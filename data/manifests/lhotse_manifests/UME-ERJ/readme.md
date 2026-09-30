## Identifers 
Recording id, supervison id, and cut id are the same string, derived from the on-disk path with `/` -> `_`. 

## lhotse fields 
`recording_id`, Uses the identifier defined above.
`start`, Set to 0.0.
`duration`, Covers the full recording duration. 
`channel`, Set to 0.
`text`, Contains the exact transcript from `doc/JEcontent/tab/*.tab` without normalization.
`language`, Set to "en".

## custom 
`custom.group`, `JE | AE | MDL`
JE: Japanese L2 speaker
AE: American English natives 
MDL: model speaker 

`custom.repeat`, bool 
`true` only for AE files whose name ends in `_R`, where the speaker produced the sentence immediately after hearing the MDL utterance rather than simply reading.

`custom.assessment`
Present on every JE supervision, so `assessment[scale] is None` is the single test for "not rated on this scale". Keys depend on `prompt.unit`  
- sentences: `segmental`, `rhythm`, `intonation`
- words: `segmental`, `accent`
Each key is either null (the file was not sampled for that scale) or a dict with all five raters: 

```json
"assessment": {
  "segmental":  {"R1": 3, "R2": 4, "R3": 3, "R4": 4, "R5": -1},
  "rhythm":     null,
  "intonation": {"R1": 2, "R2": 3, "R3": -1, "R4": -1, "R5": -1}
}
```
1-5: scores provided by the raters 
0: corpus code "cannot be rated"
-1: this rater is absent for this file 



