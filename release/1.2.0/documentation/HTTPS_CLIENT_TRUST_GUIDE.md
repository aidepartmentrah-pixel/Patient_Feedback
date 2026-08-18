# HTTPS Client Trust Guide — Patient Feedback System

This deployment serves the application over HTTPS using a **self-signed**
certificate (generated per-deployment by `scripts/generate_certificate.sh`
— there is no certificate authority involved, by design, since this server
has no internet access and no path to a publicly-trusted CA). HTTPS is
required, not optional: the Speech-to-Text microphone feature only works in
a browser secure context.

Without the steps below, every clinical workstation's browser will show a
"Your connection is not private" / "NET::ERR_CERT_AUTHORITY_INVALID"
warning on first visit. The connection is still encrypted — the warning
exists because the browser doesn't recognize who issued the certificate,
not because anything is actually wrong.

## Option 1 — click through once per browser (fastest, not centrally managed)

In the warning page: **Advanced** → **Proceed to `<server-ip>` (unsafe)**.
The browser remembers this per-site and won't ask again on that same
machine, unless the certificate is later regenerated (e.g. the server's IP
changes and `generate_certificate.sh` is re-run — every workstation would
need to click through again).

## Option 2 — install the certificate into each workstation's trust store (recommended for shared/kiosk machines)

The certificate file lives on the server at `/opt/rah/apps/pfms/certs/cert.pem`
after install. Copy it to each workstation (USB drive, or however files are
moved on this air-gapped network) and install it as a **trusted root
certificate**:

### Windows

1. Copy `cert.pem` to the workstation, rename to `pfms-cert.crt`.
2. Right-click → **Install Certificate** → **Local Machine** (needs admin) →
   **Place all certificates in the following store** → **Trusted Root
   Certification Authorities** → Finish.
3. Restart the browser.

### Linux (Debian-based, if any clinical workstations run Linux)

```bash
sudo cp pfms-cert.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

Firefox keeps its own certificate store separate from the OS on most
platforms — import it separately via **Settings → Privacy & Security →
Certificates → View Certificates → Authorities → Import**.

## If the certificate is ever regenerated

Re-running `scripts/generate_certificate.sh` (e.g. after the server's IP
changes) invalidates every workstation's prior trust decision — repeat
whichever option was used, on every workstation, again.

## Verifying the certificate matches what's actually installed

```bash
openssl x509 -in /opt/rah/apps/pfms/certs/cert.pem -noout -subject -dates -ext subjectAltName
```

Confirms the certificate's Subject Alternative Names actually include the
address being used to reach the server — a mismatch here (not a trust
issue) produces a *different* browser warning
("NET::ERR_CERT_COMMON_NAME_INVALID") and means `generate_certificate.sh`
needs to be re-run with the correct address.
