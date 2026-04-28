Requirements:

Enrich each existing song in the tracks table to include content based filtering data features

Populate the content based tables: audio_features, track_metadata, artist_metadata
	- For each track_uri in the tracks table call the API using the track_uri as the primary key to populate the audio_features table
	- For each track uri in the tracks table call the API using the track_uri as the primary key to populate the track_metadata table 
	- For each artist_uri in the tracks table call the API using artist_uri as the primary key to populate the artist_metadata table 

Update the enrichment_status in the tracks table to complete or null based on the enrichment status.
	
This allows the script to populate the feature_store table which is the join the collaborative filtering and content based filtering columns. Model ready table 

The API call:
A maximum of 50 queries can be made for each call. Three parallel queries to be made for each call.

For audio_features the API call is 
A comma-separated list of the Spotify IDs for the tracks. Maximum: 100 IDs.
Example: ids=7ouMYWpwJ422jRcDASZB7P,4VqPOruhp5EdPBeR92t6lQ,2takcwOaAZWiXQijPHIx7B

curl --request GET \
  --url 'https://api.spotify.com/v1/audio-features?ids=7ouMYWpwJ422jRcDASZB7P%2C4VqPOruhp5EdPBeR92t6lQ%2C2takcwOaAZWiXQijPHIx7B' \
  --header 'Authorization: Bearer 1POdFZRZbvb...qqillRxMr2z'

Response sample
{  "audio_features": [    {      "acousticness": 0.00242,      "analysis_url": "https://api.spotify.com/v1/audio-analysis/2takcwOaAZWiXQijPHIx7B",      "danceability": 0.585,      "duration_ms": 237040,      "energy": 0.842,      "id": "2takcwOaAZWiXQijPHIx7B",      "instrumentalness": 0.00686,      "key": 9,      "liveness": 0.0866,      "loudness": -5.883,      "mode": 0,      "speechiness": 0.0556,      "tempo": 118.211,      "time_signature": 4,      "track_href": "https://api.spotify.com/v1/tracks/2takcwOaAZWiXQijPHIx7B",      "type": "audio_features",      "uri": "spotify:track:2takcwOaAZWiXQijPHIx7B",      "valence": 0.428    }  ]


-- Audio features from GET /audio-features (batch endpoint)
CREATE TABLE IF NOT EXISTS audio_features (
    track_uri           TEXT            PRIMARY KEY REFERENCES tracks(track_uri) ON DELETE CASCADE,
    -- Perceptual high-level features
    danceability        NUMERIC(4,3),   -- 0.0–1.0
    energy              NUMERIC(4,3),
    valence             NUMERIC(4,3),
    acousticness        NUMERIC(4,3),
    instrumentalness    NUMERIC(4,3),
    liveness            NUMERIC(4,3),
    speechiness         NUMERIC(4,3),
    -- Objective sonic descriptors
    loudness            NUMERIC(6,3),   -- dB, typically -60–0
    tempo               NUMERIC(6,3),   -- BPM
    time_signature      SMALLINT,       -- beats per bar (3–7)
    key                 SMALLINT,       -- pitch class 0–11 (-1 = no key detected)
    mode                SMALLINT,       -- 0 = minor, 1 = major
    duration_ms         INT,
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);


For track_metadata the API call is

A comma-separated list of the Spotify IDs for the tracks. Maximum: 50 IDs.
Example: ids=7ouMYWpwJ422jRcDASZB7P,4VqPOruhp5EdPBeR92t6lQ,2takcwOaAZWiXQijPHIx7B

curl --request GET \
  --url 'https://api.spotify.com/v1/tracks?ids=7ouMYWpwJ422jRcDASZB7P%2C4VqPOruhp5EdPBeR92t6lQ%2C2takcwOaAZWiXQijPHIx7B' \
  --header 'Authorization: Bearer 1POdFZRZbvb...qqillRxMr2z'

Response sample
{  "tracks": [    {      "album": {        "album_type": "compilation",        "total_tracks": 9,        "available_markets": ["CA", "BR", "IT"],        "external_urls": {          "spotify": "string"        },        "href": "string",        "id": "2up3OPMp9Tb4dAKM2erWXQ",        "images": [          {            "url": "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228",            "height": 300,            "width": 300          }        ],        "name": "string",        "release_date": "1981-12",        "release_date_precision": "year",        "restrictions": {          "reason": "market"        },        "type": "album",        "uri": "spotify:album:2up3OPMp9Tb4dAKM2erWXQ",        "artists": [          {            "external_urls": {              "spotify": "string"            },            "href": "string",            "id": "string",            "name": "string",            "type": "artist",            "uri": "string"          }        ]      },      "artists": [        {          "external_urls": {            "spotify": "string"          },          "href": "string",          "id": "string",          "name": "string",          "type": "artist",          "uri": "string"        }      ],      "available_markets": ["string"],      "disc_number": 0,      "duration_ms": 0,      "explicit": false,      "external_ids": {        "isrc": "string",        "ean": "string",        "upc": "string"      },      "external_urls": {        "spotify": "string"      },      "href": "string",      "id": "string",      "is_playable": false,      "linked_from": {      },      "restrictions": {        "reason": "string"      },      "name": "string",      "popularity": 0,      "preview_url": "string",      "track_number": 0,      "type": "track",      "uri": "string",      "is_local": false    }  ]}


CREATE TABLE IF NOT EXISTS track_metadata (
    track_uri           TEXT            PRIMARY KEY REFERENCES tracks(track_uri) ON DELETE CASCADE,
    isrc                TEXT,           -- international standard recording code
    explicit            BOOLEAN,
    popularity          SMALLINT,       -- 0–100, time-decayed play count proxy
    preview_url         TEXT,
    release_date        DATE,
    release_date_precision TEXT,        -- 'year' | 'month' | 'day'
    -- Nested artist/album objects flattened
    primary_artist_id   TEXT,
    primary_artist_name TEXT,
    album_id            TEXT,
    album_name          TEXT,
    album_type          TEXT,           -- 'album' | 'single' | 'compilation'
    total_tracks        SMALLINT,       -- tracks on album
    available_markets   TEXT[],         -- array of ISO 3166-1 alpha-2 codes
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);




For the artist_metadata the API call is:

A comma-separated list of the Spotify IDs for the tracks. Maximum: 50 IDs.
Example: ids=7ouMYWpwJ422jRcDASZB7P,4VqPOruhp5EdPBeR92t6lQ,2takcwOaAZWiXQijPHIx7B

curl --request GET \
  --url 'https://api.spotify.com/v1/artists?ids=2CIMQHirSU0MQqyYHq0eOx%2C57dN52uHvrHOxijzpIgu3E%2C1vCWHaC5f2uS3yhpwWbIA6' \
  --header 'Authorization: Bearer 1POdFZRZbvb...qqillRxMr2z'


Response sample
{  "artists": [    {      "external_urls": {        "spotify": "string"      },      "followers": {        "href": "string",        "total": 0      },      "genres": ["Prog rock", "Grunge"],      "href": "string",      "id": "string",      "images": [        {          "url": "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228",          "height": 300,          "width": 300        }      ],      "name": "string",      "popularity": 0,      "type": "artist",      "uri": "string"    }  ]}


-- Artist metadata from GET /artists (batch, up to 50 per call)
CREATE TABLE IF NOT EXISTS artist_metadata (
    artist_uri          TEXT            PRIMARY KEY,
    artist_id           TEXT            GENERATED ALWAYS AS (split_part(artist_uri, ':', 3)) STORED,
    artist_name         TEXT,
    genres              TEXT[],         -- Spotify genre tags
    popularity          SMALLINT,
    followers           INT,
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

















