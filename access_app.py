"""Start the Dialygo-Access review app on this computer.

    python access_app.py                      # http://127.0.0.1:5100, this computer only
    python access_app.py --data D:\\access_data
    python access_app.py --allow-lan --host 0.0.0.0   # hospital network only, never the internet

Patient images must never go to the internet (PRD section 4): by default the app only
listens on this computer, and it refuses any other address unless --allow-lan is given.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from pathlib import Path

from access.webapp import create_app


DEFAULT_DATA = Path(__file__).resolve().parent / "access_data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("ACCESS_DATA_DIR", str(DEFAULT_DATA)), help="where the database, cache and reports live")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5100)
    ap.add_argument("--allow-lan", action="store_true", help="allow a non-local address (hospital network only)")
    a = ap.parse_args(argv)
    host_is_local = a.host in ("localhost",) or ipaddress.ip_address(a.host).is_loopback
    if not host_is_local and not a.allow_lan:
        print(f"Refusing to listen on {a.host}: this app holds patient images and runs on this computer only. "
              "Use --allow-lan only on a closed hospital network.", file=sys.stderr)
        return 2
    app = create_app(Path(a.data))
    print(f"Dialygo-Access on http://{'127.0.0.1' if host_is_local else a.host}:{a.port}  (data: {a.data})")
    try:
        from waitress import serve  # production-grade server if installed

        from access.webapp import MAX_UPLOAD_GB

        # waitress caps request bodies at 1 GB by default; studies are uploaded whole
        serve(app, host=a.host, port=a.port, threads=4, max_request_body_size=MAX_UPLOAD_GB * 1024 ** 3)
    except ImportError:
        app.run(host=a.host, port=a.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
