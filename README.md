# Incoder server

## `main` branch instructions

This branch uses sockets to communicate with the client.
The code was originally made by @mhyee, https://github.com/nuprl/TypeWeaver

## `http` branch instructions

## To run

- `python server.py`

## Example POST

- from `./test_server.py`

```py
import subprocess

API = 'http://localhost:8000'

with open('./test.ts') as f:
    input = f.read()
    n = 1
    temperature = 0.8
    stdout = subprocess.run(['curl', '-d', f'temperature={temperature}&input={input}&retries={n}', 'http://localhost:8000']).stdout
    print(str(stdout))
```
