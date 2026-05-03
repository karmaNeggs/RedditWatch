# Reddit Bot Activity Analysis System

A comprehensive, multi-faceted bot detection system for analyzing 5 Indian subreddits (r/india, r/unitedstatesofindia, r/indiaspeaks, r/teenindia, r/indiasocial).

## Features

### Four-Angle Bot Detection
1. **User-Level Indicators (35% weight)**
   - Karma-to-age ratios (suspicious: >100 karma/day)
   - Account velocity analysis
   - Very suspicious accounts (>500 karma/day)

2. **Post-Level Engagement Patterns (30% weight)**
   - Upvote-to-Comment Ratio (UCR)
   - Upvote ratio analysis (passive vs active engagement)
   - Comment-to-score relationships

3. **Temporal Patterns (20% weight)**
   - Posting time distributions
   - Peak hour concentration
   - Distribution entropy (uniformity measure)
   - Automated posting schedules detection

4. **Statistical Distribution Anomalies (15% weight)**
   - Coefficient of variation
   - Skewness analysis
   - Outlier detection (IQR method)

### Admin Dashboard
- Password-protected interface
- Manual trigger for data collection and analysis
- Real-time operation output
- Historical analysis tracking
- Download raw data and analysis results

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

1. **Clone/Download the repository**
```bash
cd reddit-bot-analysis-repo
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories**
```bash
mkdir -p data output
```

## Usage

### Option 1: Admin Dashboard (Recommended)

1. **Start the admin backend**
```bash
python3 admin-backend/app.py
```

2. **Access the dashboard**
- Open browser: http://localhost:5000
- Default password: `admin123` (CHANGE THIS!)
- Click "Start Collection" to fetch data
- Click "Start Analysis" to run analysis
- Download results as needed

### Option 2: Command Line

1. **Collect data**
```bash
python3 scripts/collect_data.py
```
This fetches 50 posts from each subreddit and analyzes top posters.

2. **Run analysis**
```bash
python3 scripts/analyze_data.py
```
This performs comprehensive multi-faceted bot detection.

## Output Files

### Data Files
- `data/reddit_data_YYYYMMDD_HHMMSS.csv` - Raw collected data
- `data/reddit_data_latest.csv` - Latest data (symlink)
- `data/metadata_latest.json` - Collection metadata

### Analysis Files
- `output/analysis_YYYYMMDD_HHMMSS.json` - Full analysis results
- `output/analysis_latest.json` - Latest analysis (symlink)

## Analysis Output

The analysis generates a unified bot activity score (0-100) for each subreddit:

- **0-30**: Minimal bot activity (organic)
- **30-50**: Low-moderate bot activity
- **50-70**: High bot activity
- **70+**: Critical bot activity

Each score includes:
- User-level bot indicators
- Engagement pattern analysis
- Temporal posting patterns
- Statistical distribution anomalies
- Converging evidence from all four angles

## Configuration

### Change Admin Password
Edit `admin-backend/app.py`:
```python
ADMIN_PASSWORD_HASH = hashlib.sha256('your_new_password'.encode()).hexdigest()
```

### Adjust Data Collection Limits
Edit `scripts/collect_data.py`:
```python
def fetch_posts(subreddit, limit=50):  # Change limit here
```

### Modify Analysis Weights
Edit `scripts/analyze_data.py`:
```python
# Component weights in calculate_unified_scores()
final_score = (
    (user_score * 0.35) +      # User-level weight
    (engagement_score * 0.30) +  # Engagement weight
    (temporal_score * 0.20) +    # Temporal weight
    (distribution_score * 0.15)  # Distribution weight
)
```

## Deployment

### Local Deployment
```bash
python3 admin-backend/app.py
```

### Server Deployment (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 admin-backend.app:app
```

### Docker Deployment
```bash
docker build -t reddit-bot-analysis .
docker run -p 5000:5000 reddit-bot-analysis
```

## API Endpoints

All endpoints require authentication (login first).

- `POST /api/login` - Authenticate with password
- `POST /api/logout` - Logout
- `GET /api/status` - Get current system status
- `POST /api/collect-data` - Trigger data collection
- `POST /api/run-analysis` - Trigger analysis
- `GET /api/latest-analysis` - Get latest analysis results
- `GET /api/download-analysis` - Download analysis as JSON
- `GET /api/download-data` - Download data as CSV
- `GET /api/history` - Get analysis history

## Understanding the Analysis

### Bot Activity Score Components

1. **User-Level Score**
   - % of suspicious accounts (>100 karma/day)
   - Average karma per day
   - Presence of very suspicious accounts (>500 karma/day)

2. **Engagement Score**
   - High UCR (>20) = passive engagement (bot-like)
   - High upvote ratio (>95%) = consensus/coordinated
   - Low comments relative to score = less discussion

3. **Temporal Score**
   - High concentration in top 3 hours = automated
   - Low entropy = predictable posting schedule
   - Few hours with posts = concentrated activity

4. **Distribution Score**
   - High coefficient of variation = uneven distribution
   - High skewness = extreme values
   - Many outliers = anomalous activity

### Interpretation Guide

**r/indiaspeaks (High Bot Activity)**
- 50%+ suspicious users
- 27+ UCR (very passive)
- 38%+ posts in top 3 hours
- Converging evidence across all angles

**r/india (Minimal Bot Activity)**
- <15% suspicious users
- 12-13 UCR (balanced engagement)
- 30-36% posts in top 3 hours
- Organic user base and engagement

## Troubleshooting

### Data Collection Fails
- Check internet connection
- Reddit API may be rate-limiting (wait 5-10 minutes)
- Ensure you have valid Python environment

### Analysis Errors
- Ensure data files exist in `data/` directory
- Check that `data/reddit_data_latest.csv` is present
- Verify all required Python packages are installed

### Dashboard Won't Start
- Check if port 5000 is already in use
- Try: `python3 admin-backend/app.py --port 5001`
- Ensure Flask and dependencies are installed

## Security Notes

1. **Change default password** before deployment
2. **Use HTTPS** in production
3. **Restrict IP access** to admin dashboard
4. **Use environment variables** for sensitive data
5. **Regularly backup** data and analysis files

## Contributing

To improve the analysis:
1. Modify detection algorithms in `scripts/analyze_data.py`
2. Adjust weights for different components
3. Add new analysis angles
4. Improve data collection robustness

## License

MIT License

## Support

For issues or questions, refer to the documentation or create an issue.

---

**Last Updated**: February 2026
**Version**: 1.0
