import os
import sys
import json
import torch
import socket
import signal
import argparse
from threading import Thread
from functools import partial

import model
from infer import infer

from typing import List

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket_path", type=str, help="The path to the socket")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="The device for the model")
    parser.add_argument("--max_length", type=int, default=2048, help="The max length for the context window")
    args = parser.parse_args()
    return args

args = get_args()

BUFF_SIZE = 4096

# checks if in use
try:
    os.unlink(args.socket_path)
except OSError:
    if os.path.exists(args.socket_path):
        print(f'{args.socket_path} already exists')
        sys.exit(1)

# used to store and close all sockets before exit
class SocketManager:
    def __init__(self) -> None:
        self._sockets = set()

    def __call__(self, c: socket.socket) -> None:
        self._sockets.add(c)

    def close_all(self) -> None:
        while len(self._sockets) > 0:
            s = self._sockets.pop()
            s.close()

# an unbounded recv
def recvall(s: socket.socket) -> bytes:
    data = b''
    while True:
        part = s.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            break
    return data

END_TOKEN = "??END??"
# handles a single client
def on_client(c: socket.socket) -> None:
    try:
        complete_data = ""
        while True:
            data = recvall(c).decode('utf-8')
            if len(data) == 0:
                break
            if not data.endswith(END_TOKEN):
                complete_data += data
                continue
            complete_data += data[:-len(END_TOKEN)]

            req = json.loads(complete_data)
            print(f"Received {req}")
            code = req["code"]
            num_comps = req["num_samples"]
            temperature = req["temperature"]
            type_annotations: List[str] = infer(
                model_dict=model_dict,
                code=code,
                num_comps=num_comps,
                temperature=temperature,
            )
            print(f'Result: {type_annotations}')
            resp = json.dumps({
                "type": "single",
                'type_annotations': [item for item in type_annotations]
            }).encode("utf-8") # [Vec<String>]
            print(f'Sending {resp}')
            c.sendall(resp)
    finally:
        c.close()

# listen for clients
def init_wait(s: socket.socket, sm: SocketManager) -> None:
    while True:
        c, _ = s.accept()
        sm(c)
        thread = Thread(target=on_client, args=(c,))
        thread.start()

# called on exit signal
def close(_, __, sm: SocketManager) -> None:
    print(f'Closing {args.socket_path}')
    sm.close_all()
    sys.exit(0)

# load model on device

# load model on device
print(f'Loading model on device: `{args.device}`')
if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda", int(args.device))
model_dict = model.init_model("facebook/incoder-6B", device=args.device)

# init socket manager
sm = SocketManager()
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(args.socket_path)
sock.listen(1)
# store socket for future close
sm(sock)

# this should work but should be tested
# other way is to use a lambdas
signal.signal(signal.SIGINT, partial(close, sm)) # type: ignore
print(f'Listening on {args.socket_path}\n', flush=True)
init_wait(sock, sm)
