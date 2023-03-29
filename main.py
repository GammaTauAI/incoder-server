import os
import sys
import json
import base64
import socket
import signal
from threading import Thread
from functools import partial

from infill import infill

assert len(sys.argv) == 2
SOCKET_PATH = sys.argv[1]
BUFF_SIZE = 4096

# checks if in use
try:
    os.unlink(SOCKET_PATH)
except OSError:
    if os.path.exists(SOCKET_PATH):
        print(f'{SOCKET_PATH} already exists')
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

# handles a single client
def on_client(c: socket.socket) -> None:
    try:
        while True:
            data = recvall(c)
            req = json.loads(data)
            code = str(base64.b64decode(req.code))
            num_samples = req.num_samples
            should_infill_single = req.should_infill_single
            type_annotations = infill(code, num_samples, should_infill_single)
            if should_infill_single:
                c.send(json.dumps({
                    "type": "single",
                    'type_annotations': [base64.b64decode(item) for item in type_annotations]
                }).encode("utf-8")) # [Vec<String>]
            else:
                # decode each item in the 2d array
                c.send(json.dumps({
                    "type": "multiple",
                    'type_annotations': [
                        [[base64.b64decode(item) for item in row] for row in type_annotations]
                    ]
                }).encode("utf-8")) # Vec<Vec<String>>
    finally:
        c.close()

# listen for clients
def init_wait(s: socket.socket, sm: SocketManager) -> None:
    while True:
        c, _ = s.accept()
        sm(c)
        Thread(target=on_client, args=(c,))

# called on exit signal
def close(_, __, sm: SocketManager) -> None:
    print(f'Closing {SOCKET_PATH}')
    sm.close_all()
    sys.exit(0)

# init socket manager
sm = SocketManager()
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(SOCKET_PATH)
sock.listen(1)
# store socket for future close
sm(sock)

# this should work but should be tested
# other way is to use a lambdas
signal.signal(signal.SIGINT, partial(close, sm)) # type: ignore
print(f'Listening on {SOCKET_PATH}\n')
init_wait(sock, sm)
