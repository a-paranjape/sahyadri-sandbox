#!/usr/bin/env python3
"""HTTP proxy that emulates a wide-area link for parquet_http_test.py: each request waits RTT seconds before it is
forwarded (one round trip), and the response is sent at most BW bytes/s. Range and HEAD requests pass through.
Connection set-up (TCP/TLS handshakes) is not emulated, so real links are somewhat slower for few, small requests.

usage: wan_proxy.py <listen port> <upstream base URL> <rtt s> <bandwidth MB/s>
"""
import http.server
import socketserver
import sys
import time
import urllib.error
import urllib.request

PORT, UP, RTT, BW = int(sys.argv[1]), sys.argv[2].rstrip('/'), float(sys.argv[3]), float(sys.argv[4]) * 1e6


class H(http.server.BaseHTTPRequestHandler):
    """HTTP proxy to ``UP`` that adds ``RTT`` seconds per request (simulates a wide-area link)."""
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        """Silence the default request log."""
        pass

    def _fwd(self, method):
        """Forward the request (with its Range header) after the delay and relay status, headers and body."""
        time.sleep(RTT)
        req = urllib.request.Request(UP + self.path, method=method)
        if self.headers.get('Range'):
            req.add_header('Range', self.headers['Range'])
        try:
            r = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            r = e
        self.send_response(r.status)
        for k in ('Content-Length', 'Content-Range', 'Accept-Ranges', 'Last-Modified'):
            if r.headers.get(k):
                self.send_header(k, r.headers[k])
        self.end_headers()
        if method == 'GET':
            t0, sent = time.time(), 0
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                self.wfile.write(b)
                sent += len(b)
                ahead = sent / BW - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)

    def do_GET(self):
        """Forward a GET."""
        self._fwd('GET')

    def do_HEAD(self):
        """Forward a HEAD."""
        self._fwd('HEAD')


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded HTTP server (daemon threads, address reuse)."""
    daemon_threads = True
    allow_reuse_address = True


if __name__ == '__main__':
    S(('127.0.0.1', PORT), H).serve_forever()
