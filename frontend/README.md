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
            {"start": 0, "end": 3}, 
            {"start": 5, "end": 6}, 
            ...
        ]
    }
}
```
**Align by character**