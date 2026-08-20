# Security Policy

## Supported Versions

Security fixes are provided for the latest Celiums BitNet release. Development
snapshots are supported on a best-effort basis.

## Reporting A Vulnerability

Do not disclose vulnerabilities in a public issue. Use private vulnerability
reporting on the Celiums BitNet repository when it is available, or contact
Celiums Solutions LLC through the private security channel listed on the
Celiums organization profile.

Include the affected version or commit, configuration, reproduction steps,
impact, and any proof of concept that can be shared safely.

If the problem also affects Microsoft BitNet, llama.cpp, or another dependency,
Celiums will coordinate disclosure with the relevant upstream project. Issues
that only affect this fork should not be sent to the Microsoft Security Response
Center.

## Deployment Guidance

- `celiums-bitnet serve` and `celiums-runtime-server` reject non-loopback binds
  unless an API key is configured or `--allow-unauthenticated-remote` is passed.
  `CELIUMS_BITNET_API_KEY` is canonical. `LLAMA_API_KEY` is accepted for
  migration, but now enforces authentication rather than repeating the unsafe
  0.2.0 precheck. Authentication covers every HTTP endpoint.
- Do not expose the server to untrusted networks without authentication,
  network isolation, TLS termination, and request limits.
- Treat model and adapter files as untrusted input.
- Pin model revisions and verify checksums in production.
- Run inference services with the least privileges they require.
