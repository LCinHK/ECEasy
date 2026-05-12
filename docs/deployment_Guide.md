# ECEasy Deployment Guide (Instant Version)

## Goal
Deploy a lightweight public ECEasy service quickly on a Linux VPS using your existing stack (Apache + Python), without a full account system.

**⚠️ Security Note:** Before deployment, review `DEPLOYMENT_CHECKLIST.md` for essential permission settings (`chmod 600` for `.env`, directory permissions, and system hardening).

## Recommended Architecture
- Apache serves static UI and terminates HTTPS.
- Apache reverse-proxies backend API traffic to FastAPI (`uvicorn`) on localhost.
- FastAPI runs under `systemd` for auto-restart.
- Knowledge/index files stay on disk (FAISS/Chroma), backed up regularly.

## App-Level Runtime Controls
Use `.env` for operational switches:
- `UI_VERSION` to choose old/new UI build.
- `KNOWLEDGE` to choose `Faiss` or `Chroma`.
- host/port and backend provider settings.
- server-owned API keys and default model.

**CRITICAL:** Protect `.env` with `chmod 600` (see `DEPLOYMENT_CHECKLIST.md` § 1).

## Minimal Secure Baseline
- Keep secrets only in `.env` (never commit).
- Restrict `.env` file permissions with `chmod 600` (owner read-write only).
- Secure all directories and files with appropriate ownership and permissions.
- Run service as non-root `systemd` user.
- Restrict CORS to your domain.
- Limit request size and add basic rate limiting.
- Enable HTTPS and redirect HTTP to HTTPS.
- Rotate logs and keep error logs.

## Deployment Steps (High Level)
1. Build frontend (`npm i`, `npm run build`) in selected UI directory.
2. Place build artifacts in deployed static path (`newUI` and/or `ui`).
3. Create Python virtualenv and install backend requirements.
4. **Apply security permissions** (see `DEPLOYMENT_CHECKLIST.md`).
5. Configure `systemd` service for `uvicorn`.
6. Configure Apache virtual host with reverse proxy to backend.
7. Point domain DNS to VPS and issue TLS cert.
8. Validate security checklist before final launch.
9. Run smoke tests for streaming response, sources, and image links.

## Session/History (Current Behavior)
- Without user accounts, history is browser-scoped.
- In the updated new UI, chat threads can be persisted in browser storage.
- This allows multi-chat history per browser/device without login.

## Backup Plan
Back up:
- `.env`
- FAISS/Chroma index directories
- deployment configs (`systemd` + Apache vhost)
- optional exported docs/knowledge source manifests

**Note:** Restrict backup files with `chmod 700` (see `DEPLOYMENT_CHECKLIST.md` § 10).

## Roadmap After Instant Launch
1. Harden operations (monitoring, health checks, alerts).
2. Add optional anonymous session IDs on backend.
3. Add account system (OAuth/email), then move chat history server-side.
4. Add per-user quotas and usage analytics.

---

## Security Checklist Quick Reference

| Task | Command |
|------|---------|
| Restrict `.env` | `chmod 600 .env` |
| Secure directories | `chmod 750 /path/to/eceasy` |
| Restrict logs | `chmod 750 /path/to/eceasy/logs` |
| Set ownership | `sudo chown -R eceasy:eceasy /path/to/eceasy` |
| Verify permissions | `find /path -type f -perm /077 -ls` |

**👉 Full checklist:** See `DEPLOYMENT_CHECKLIST.md`
