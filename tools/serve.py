# Static file server for local development: `npm run serve` (port 8002, or the first argument).
#
# python3 -m http.server listens with a backlog of 5 connections. The page loads some 40 ES modules at once, and
# when the machine is busy (the test suite running beside a page check) the server falls behind, the queue
# overflows and the browser gets connection resets: modules fail to load and the app never starts. A burst test
# (60 requests at once, 300 times) reset about a third of them with a backlog of 5 and none with 128.
import functools
import http.server
import socket
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
BACKLOG = 128  # connections waiting to be accepted


class Server(http.server.ThreadingHTTPServer):
    # IPv6 with IPV6_V6ONLY off takes IPv4 too, as python3 -m http.server does, so both localhost (::1)
    # and 127.0.0.1 reach it
    address_family = socket.AF_INET6
    request_queue_size = BACKLOG

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory='.')
with Server(('::', PORT), handler) as server:
    print(f'Serving on http://localhost:{PORT}/')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
