"""Read-only checks for the independently started delivery container."""
import json
import re
import urllib.request

BASE = 'http://127.0.0.1:19003'

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return response.status, response.read()

if __name__ == '__main__':
    code, health = get('/health')
    assert code == 200
    print('health:', health.decode())
    code, html = get('/')
    assert code == 200 and b'<html' in html.lower()
    assets = re.findall(rb'(?:src|href)="(/assets/[^\"]+)"', html)
    assert assets, 'No compiled frontend assets found'
    for asset in assets:
        code, data = get(asset.decode())
        assert code == 200 and data
    print(json.dumps({'spa': 'PASS', 'compiled_assets': len(assets)}, ensure_ascii=False))
    from pathlib import Path
    password_file = Path('/data/evalperf-deploy/acceptance-admin-password')
    if password_file.exists():
        payload = json.dumps({'username': 'admin', 'password': password_file.read_text().strip()}).encode()
        request = urllib.request.Request(BASE + '/api/v1/auth/login', data=payload,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
        assert result['user']['role'] == 'admin' and result['token']
        print('initial admin login: PASS')
