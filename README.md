# The InCoder Server for OpenTau

A socket for the Rust client in [OpenTau](https://github.com/GammaTauAI/opentau) for type-inference using [InCoder](https://huggingface.co/facebook/incoder-6B).

The code was originally written by @arjunguha and later adapted by @mhyee, https://github.com/nuprl/TypeWeaver

### To run independently:
  - clone this repo
```bash
git clone https://github.com/GammaTauAI/incoder-server && cd incoder-server
```
  - install dependencies
```bash
pip install -r ./requirements.txt
```
  - start the socket
```bash
python ./main.py <socket path>
```
  - test the socket with a request
```bash
python ./test_socket.py <socket path>
```
