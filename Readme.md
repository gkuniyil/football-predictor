# League Predictor & Season Simulator — Day 1 Setup

## What's here
- `docker-compose.yml` — spins up Postgres + Redis
- `src/schema.sql` — the 3-table schema (teams, matches, team_ratings), auto-runs on first Postgres startup
- `requirements.txt` — Python deps
- `.env.example` — copy to `.env` and fill in your API key

## Manual steps (do these now)

1. **RapidAPI account + API-Football key**
   - Go to rapidapi.com, sign up, search "API-Football" (by API-Sports), subscribe to the free tier.
   - Copy your API key into `.env` as `API_FOOTBALL_KEY`.

2. **Kaggle dataset**
   - Search Kaggle for "La Liga results" or "Spanish football results" — look for one covering at least the last 5 seasons.
   - Download the CSV, drop it in `data/` (e.g. `data/historical_matches.csv`).

3. **Local setup**
   ```bash
   cd league-predictor
   cp .env.example .env          # then edit .env with your real API key
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   docker compose up -d          # starts Postgres + Redis, auto-creates schema.sql tables
   ```

4. **Verify it worked**
   ```bash
   docker exec -it league_predictor_db psql -U league_admin -d league_predictor -c "\dt"
   ```
   You should see `teams`, `matches`, `team_ratings` listed.

## Next (Day 1, Step 2)
Once Postgres is up and the Kaggle CSV is downloaded, next step is writing the ingestion script that loads the CSV into `teams`/`matches` and pulls current-season fixtures from API-Football.