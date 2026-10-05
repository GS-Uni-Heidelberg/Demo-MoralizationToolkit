# Safety Todo List

This checklist records the website safety review and the remediation work.

## High priority

- [x] Reserve batch credits before work starts so rejected requests cannot obtain results.
- [ ] Refund reserved credits when a failed job cannot complete.
- [x] Make admin authentication fail closed when `ADMIN_API_KEY` is missing; refuse production startup without it.
- [x] Remove the lemmatizer XSS sink. Do not render raw user text with `dangerouslySetInnerHTML`; escape text or render marked fragments as React elements.
- [x] Enforce a 50 MB upload limit before parsing and validate every lemma-batch text length.
- [ ] Use a bounded worker queue and expire completed jobs.

## Medium priority

- [x] Bind batch and lemma jobs to a separate random capability token; require it for status and result access.
- [x] Replace persistent browser `localStorage` API tokens with `sessionStorage`.
- [ ] Prefer HttpOnly, secure, short-lived credentials for stronger XSS resistance.
- [x] Configure the API origin through `NEXT_PUBLIC_API_BASE_URL`; deploy it with an HTTPS value.
- [x] Add an in-process per-IP rate limit for expensive inference and lemmatizer endpoints. Add proxy/shared-store limits for multi-worker production deployments.
- [x] Avoid exposing internal exception text and filesystem details through `/health` and generic 500 responses.

## Dependencies and deployment

- [x] Upgrade vulnerable production frontend dependencies reported by `npm audit`, including Next.js and the Plotly/MapLibre dependency chain.
- [x] Run a Python dependency audit with `pip-audit` or an equivalent scanner.
- [x] Pin and use Node.js `22.23.2` (which satisfies the declared Next.js requirement and the upgraded Plotly engine).
- [x] Fix the existing frontend lint errors and make the production build pass.
- [x] Keep secrets, the billing database, model files, and generated build output out of version control.

## Review evidence

- Backend syntax validation passed during the initial review.
- The local admin endpoint rejected an unauthenticated request because `ADMIN_API_KEY` is currently configured locally.
- `npm audit --omit=dev` reports 0 vulnerabilities after the dependency upgrades.
- Full `npm audit` still reports 5 high-severity dev-only findings in `eslint-config-next`'s transitive `braces` chain; npm only offers an incompatible downgrade to `eslint-config-next@14.2.35`.
- `npm run lint` passes under Node `22.23.2`.
- `npm run build` passes under Node `22.23.2`.
- `pip-audit -r backend/requirements.txt` reports no known vulnerabilities.