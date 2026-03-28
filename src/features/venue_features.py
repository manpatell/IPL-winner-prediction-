"""
Venue-level features:
  - Average first-innings score at venue (pitch type proxy)
  - Toss impact at venue: does winning toss + choosing field win more often here?
  - Boundary size encoding (small=batting friendly, large=bowling friendly)
"""

# Average first-innings score based on IPL statistics (real data)
VENUE_AVG_SCORE = {
    "Wankhede Stadium":                   182,  # Mumbai — small ground, batting paradise
    "M Chinnaswamy Stadium":              178,  # Bangalore — high altitude, small boundary
    "MA Chidambaram Stadium":             163,  # Chennai — slow pitch, spin-friendly
    "Eden Gardens":                       162,  # Kolkata — dew factor, balanced
    "Narendra Modi Stadium":              168,  # Ahmedabad — large ground, balanced
    "Rajiv Gandhi Intl Stadium":          173,  # Hyderabad — batting friendly
    "Punjab Cricket Association Stadium": 172,  # Mohali — fast outfield
    "Sawai Mansingh Stadium":             167,  # Jaipur — spin helps
    "Feroz Shah Kotla":                   156,  # Delhi — slow pitch
    "BRSABV Ekana Cricket Stadium":       165,  # Lucknow — balanced
    "DY Patil Stadium":                   175,  # Mumbai — batting
    "Brabourne Stadium":                  170,  # Mumbai — batting
    "Maharashtra Cricket Association Stadium": 164,  # Pune — balanced
    "Dubai Intl Stadium":                 158,  # UAE — slow pitch
    "Sharjah Cricket Stadium":            162,  # UAE — small ground
    "Abu Dhabi Cricket Ground":           155,  # UAE — slow
    "Newlands":                           156,
    "Kingsmead":                          162,
    "New Wanderers":                      170,
    "SuperSport Park":                    165,
    "Centurion":                          165,
}

# Toss impact at venue: probability that toss winner wins (1 = big impact, 0.5 = none)
# Venues with dew factor → fielding second strongly preferred → toss matters more
VENUE_TOSS_IMPACT = {
    "Wankhede Stadium":                   0.57,  # Dew factor in evenings
    "M Chinnaswamy Stadium":              0.56,  # High dew
    "MA Chidambaram Stadium":             0.52,  # Less dew
    "Eden Gardens":                       0.58,  # Heavy dew in Kolkata
    "Narendra Modi Stadium":              0.54,
    "Rajiv Gandhi Intl Stadium":          0.55,
    "Punjab Cricket Association Stadium": 0.54,
    "Sawai Mansingh Stadium":             0.53,
    "Feroz Shah Kotla":                   0.55,  # Dew in Delhi
    "BRSABV Ekana Cricket Stadium":       0.55,
    "DY Patil Stadium":                   0.56,
    "Brabourne Stadium":                  0.55,
    "Dubai Intl Stadium":                 0.54,
    "Sharjah Cricket Stadium":            0.53,
    "Abu Dhabi Cricket Ground":           0.52,
}

# Ground size category: 0 = small (batting paradise), 1 = medium, 2 = large
VENUE_SIZE = {
    "Wankhede Stadium":                   0,
    "M Chinnaswamy Stadium":              0,
    "Sharjah Cricket Stadium":            0,
    "MA Chidambaram Stadium":             1,
    "Eden Gardens":                       1,
    "Rajiv Gandhi Intl Stadium":          1,
    "Punjab Cricket Association Stadium": 1,
    "Sawai Mansingh Stadium":             1,
    "BRSABV Ekana Cricket Stadium":       1,
    "DY Patil Stadium":                   1,
    "Feroz Shah Kotla":                   1,
    "Brabourne Stadium":                  1,
    "Narendra Modi Stadium":              2,   # Largest cricket stadium
    "Dubai Intl Stadium":                 1,
    "Abu Dhabi Cricket Ground":           1,
}

GLOBAL_AVG_SCORE   = 167   # IPL overall average
GLOBAL_TOSS_IMPACT = 0.54  # IPL average toss impact


def get_venue_avg_score(venue: str) -> float:
    """Normalized venue average score (0–1, where 1 = highest scoring)."""
    raw = VENUE_AVG_SCORE.get(venue, GLOBAL_AVG_SCORE)
    # Normalize to 0–1 range (140=0, 195=1)
    return (raw - 140) / (195 - 140)


def get_venue_toss_impact(venue: str) -> float:
    """Returns toss win probability at this venue (>0.5 = toss matters more)."""
    return VENUE_TOSS_IMPACT.get(venue, GLOBAL_TOSS_IMPACT)


def get_venue_size(venue: str) -> int:
    """0=small, 1=medium, 2=large."""
    return VENUE_SIZE.get(venue, 1)
