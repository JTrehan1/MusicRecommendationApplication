# MusicRecommendationApplication

# Architecture Note — Content-Based Filtering

The original hybrid architecture planned a CBF layer using Spotify Web API
data (audio features, track metadata, artist metadata). This was deprecated
in 2025 when Spotify's Developer Terms were updated to explicitly prohibit
using their platform data for ML model training.

The `feature/data-pipelines-enrichment` branch preserves this architecture for reference.
The production model uses collaborative filtering only, sourced entirely
from the MPD dataset.

ML application to create song suggestions for individual users they can add to playlists or have on continuous shuffle. 
