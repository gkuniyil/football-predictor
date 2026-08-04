-- League Predictor: core schema
-- Starting with ONE league (La Liga) for the first version.

CREATE TABLE teams (
    team_id     SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    league_id   VARCHAR(50) NOT NULL DEFAULT 'la_liga'
);

CREATE TABLE matches (
    match_id       SERIAL PRIMARY KEY,
    date           DATE NOT NULL,
    home_team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals     INTEGER,
    away_goals     INTEGER,
    result         CHAR(1),  -- 'H', 'D', 'A' -- null until match is played
    competition    VARCHAR(50) NOT NULL DEFAULT 'la_liga',
    CONSTRAINT different_teams CHECK (home_team_id != away_team_id)
);

CREATE TABLE team_ratings (
    rating_id    SERIAL PRIMARY KEY,
    team_id      INTEGER NOT NULL REFERENCES teams(team_id),
    date         DATE NOT NULL,
    elo_rating   NUMERIC(7,2) NOT NULL,
    UNIQUE (team_id, date)
);

-- Index for the as-of-date lookups you'll do constantly during feature engineering
CREATE INDEX idx_matches_date ON matches(date);
CREATE INDEX idx_ratings_team_date ON team_ratings(team_id, date);