# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Data Directories
```bash
mkdir -p data output
```

### 3. Start Admin Dashboard
```bash
python3 admin-backend/app.py
```

### 4. Access Dashboard
- Open: http://localhost:5000
- Password: `admin123`

### 5. Run Analysis
1. Click "Start Collection" (takes 10-15 minutes)
2. Click "Start Analysis" (takes 1-2 minutes)
3. View results in dashboard
4. Download data/analysis as needed

---

## Manual Command Line Usage

### Collect Data Only
```bash
python3 scripts/collect_data.py
```
Output: `data/reddit_data_latest.csv`

### Run Analysis Only
```bash
python3 scripts/analyze_data.py
```
Output: `output/analysis_latest.json`

### View Results
```bash
cat output/analysis_latest.json | python3 -m json.tool
```

---

## Understanding Results

### Bot Activity Score
- **0-30**: Minimal bots (organic)
- **30-50**: Low-moderate bots
- **50-70**: High bot activity
- **70+**: Critical bot activity

### Example Output
```json
{
  "indiaspeaks": {
    "final_score": 52.9,
    "user_score": 53.3,
    "engagement_score": 64.1,
    "temporal_score": 66.3,
    "distribution_score": 11.3
  }
}
```

---

## Important Notes

1. **First run takes time**: Data collection takes 10-15 minutes (Reddit API rate limiting)
2. **Change password**: Edit `admin-backend/app.py` before production use
3. **Backup results**: Keep copies of analysis files
4. **Network required**: Both data collection and analysis need internet

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 5000 in use | Change port: `python3 admin-backend/app.py --port 5001` |
| Data collection fails | Wait 5-10 minutes, Reddit may be rate-limiting |
| No analysis results | Ensure data collection completed first |
| Dashboard won't load | Check Flask is running, refresh browser |

---

## Next Steps

1. **Customize analysis**: Edit weights in `scripts/analyze_data.py`
2. **Deploy online**: Use Gunicorn or Docker
3. **Schedule runs**: Set up cron job or task scheduler
4. **Integrate results**: Build custom visualizations with analysis JSON

---

**Need help?** See README.md for detailed documentation.
