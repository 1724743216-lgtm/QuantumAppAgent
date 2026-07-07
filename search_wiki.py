import urllib.request, ssl, json, sys

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

page = sys.argv[1] if len(sys.argv) > 1 else "Pritzker_School_of_Molecular_Engineering"
url = f"https://en.wikipedia.org/w/api.php?action=parse&page={page}&prop=wikitext&format=json"
req = urllib.request.Request(url, headers={"User-Agent": "QuantumResearch/1.0"})
resp = urllib.request.urlopen(req, timeout=15, context=ctx)
d = json.loads(resp.read().decode('utf-8'))
text = d['parse']['wikitext']['*']
lines = text.split('\n')
for line in lines:
    if 'quantum' in line.lower() and any(kw in line.lower() for kw in ['degree', 'program', 'phd', 'master', 'bachelor']):
        print(line.strip()[:400])
