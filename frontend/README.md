## POST REQUEST 
```
{
    "text":  "hôm nay toi dihoc . cjpcsong rất là vui",
    "model": 0
}
```
Model type:
- Tokenization repair: 0
- Corrector: 1
- Tokenization repair + corrector: 2

Response should contain 
```
{
    "result": {
        "text": corrected text,
        "align": [
            {"start": i_1, "end": j_1}, 
            {"start": i_2, "end": j_2}, 
            ...
        ]
    }
}
```