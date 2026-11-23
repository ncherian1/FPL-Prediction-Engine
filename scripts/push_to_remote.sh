#!/usr/bin/env bash
# Copy the local docker Postgres warehouse into a hosted Postgres (e.g. Supabase).
#   scripts/push_to_remote.sh "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres"
# Use the Supabase *Session pooler* URL (port 5432). Requires `docker compose up` locally.
set -euo pipefail
REMOTE_URL="${1:?usage: $0 <remote DATABASE_URL>}"
cd "$(dirname "$0")/.."

echo "1/3 creating schema on remote"
DATABASE_URL="$REMOTE_URL" .venv/bin/python -m fpl_engine migrate

echo "2/3 dumping local data"
DUMP=$(mktemp -d)/fpl.dump
docker compose exec -T db pg_dump -U fpl -d fpl --data-only --format=custom \
  --schema=raw --schema=core --schema=ops --exclude-table=ops.schema_migrations > "$DUMP"
ls -lh "$DUMP"

echo "3/3 restoring into remote (postgres:16 client in docker)"
docker run --rm -i --network host -v "$(dirname "$DUMP")":/dump postgres:16 \
  pg_restore --data-only --no-owner --no-privileges \
  --single-transaction -d "$REMOTE_URL" /dump/fpl.dump

echo "done — row counts on remote:"
DATABASE_URL="$REMOTE_URL" .venv/bin/python - <<'PY'
from fpl_engine import db
with db.connect() as c:
    for t in ["core.players","core.fixtures","core.player_match_stats","raw.football_data_matches","raw.provider_records"]:
        print(f"  {t}: {c.execute(f'select count(*) from {t}').fetchone()[0]}")
PY
