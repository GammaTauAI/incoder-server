import json
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = '127.0.0.0'
PORT = 8000


class IncoderServer(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        data = {
            'success': True
        }
        self.wfile.write(json.dumps(data).encode('utf-8'))

def run() -> None:
    server = HTTPServer((HOST, PORT), IncoderServer)
    print('incoder server is now running!')
    server.serve_forever()
    server.server_close()

if __name__ == '__main__':
    run()
