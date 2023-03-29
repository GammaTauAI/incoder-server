import sys
import json
import socket

assert len(sys.argv) == 2
SOCKET_PATH = sys.argv[1]

unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
unix_socket.connect(SOCKET_PATH)
payload = {
        "code": "function add(a: _hole_, b: _hole_) { return a + b }",
        "num_samples": 1,
        "should_sample_single": True,
}
unix_socket.sendall(json.dumps(payload).encode("utf-8"))

response = unix_socket.recv(1024)
print(response.decode())
unix_socket.close()
