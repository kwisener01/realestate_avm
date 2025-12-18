# Railway Deployment Guide

This guide walks you through deploying your Real Estate AVM app to Railway with a user-friendly web interface.

## What You'll Get

- **Public URL**: `https://your-app.railway.app`
- **Web UI**: Simple interface for users to submit Google Sheets URLs
- **API Docs**: Interactive API documentation at `/docs`
- **Auto-deploy**: Automatic deployments on every git push
- **Free Tier**: $5/month credit (enough for development/testing)

---

## Prerequisites

1. **GitHub Account** - Your code needs to be on GitHub
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Google Service Account** - For Google Sheets API access

---

## Step 1: Prepare Your Code

### 1.1 Commit Current Changes

```bash
# Activate virtual environment
source venv/Scripts/activate

# Stage all files
git add .

# Commit
git commit -m "feat: add Google Sheets integration with web UI"

# Push to GitHub
git push origin master
```

### 1.2 Verify Files Are Ready

Ensure these files exist (they should be created already):
- ✅ `Procfile` - Tells Railway how to start the app
- ✅ `requirements.txt` - Python dependencies
- ✅ `runtime.txt` - Python version
- ✅ `railway.json` - Railway configuration
- ✅ `app/static/index.html` - Web UI

---

## Step 2: Create Google Service Account

### 2.1 Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project: "Real Estate AVM"
3. Enable APIs:
   - Google Sheets API
   - Google Drive API

### 2.2 Create Service Account

1. Go to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Name: `realestate-avm-bot`
4. Grant role: "Editor"
5. Click "Create Key" → JSON format
6. Save the downloaded file as `credentials.json`

### 2.3 Note the Service Account Email

In the JSON file, find:
```json
{
  "client_email": "realestate-avm-bot@xxx.iam.gserviceaccount.com"
}
```

You'll need this email to share your Google Sheets!

---

## Step 3: Deploy to Railway

### Option A: Via Railway Dashboard (Easiest)

1. **Go to [railway.app](https://railway.app) and login**

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your GitHub
   - Select your `realestate_avm` repository

3. **Configure Environment Variables**
   - Click on your deployed service
   - Go to "Variables" tab
   - Click "Raw Editor"
   - Paste your entire `credentials.json` content:

   ```
   GOOGLE_SHEETS_CREDENTIALS={"type":"service_account","project_id":"...","private_key":"..."}
   ```

   **Important**: Copy the entire JSON content as a single line (no newlines except within the private_key string)

   Alternative: Create the variable as a multiline secret:
   - Click "+ New Variable"
   - Name: `GOOGLE_SHEETS_CREDENTIALS`
   - Value: Paste entire JSON file content

4. **Deploy**
   - Railway will automatically build and deploy
   - Wait 3-5 minutes for first deployment
   - You'll get a URL like: `https://realestate-avm-production.up.railway.app`

5. **Enable Public Access**
   - Go to "Settings" tab
   - Find "Networking" section
   - Click "Generate Domain"
   - Copy your public URL

### Option B: Via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Set environment variable
railway variables set GOOGLE_SHEETS_CREDENTIALS="$(cat credentials.json | tr -d '\n')"

# Deploy
railway up
```

---

## Step 4: Test Your Deployment

### 4.1 Check Health

Visit: `https://your-app.railway.app/health`

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": {...}
}
```

### 4.2 Test Google Sheets Integration

Visit: `https://your-app.railway.app/sheets/health`

Expected:
```json
{
  "status": "ready",
  "credentials_configured": true,
  "model_loaded": true
}
```

### 4.3 Access Web UI

Visit: `https://your-app.railway.app`

You should see the property valuation web interface!

---

## Step 5: Use the App

### 5.1 Prepare Your Google Sheet

1. Create a Google Sheet with required columns (see `GOOGLE_SHEETS_INTEGRATION.md`)
2. Click "Share" button
3. Add the service account email: `realestate-avm-bot@xxx.iam.gserviceaccount.com`
4. Grant "Editor" access
5. Copy the sheet URL

### 5.2 Submit for Analysis

1. Open your deployed app: `https://your-app.railway.app`
2. Paste your Google Sheets URL
3. Configure options (write back, ensemble model)
4. Click "Analyze Properties"
5. View results in real-time!

### 5.3 Check Results in Sheet

If you enabled "Write back", check columns N, O, P in your sheet:
- **Column N**: predicted_price
- **Column O**: confidence
- **Column P**: timestamp

---

## Step 6: Set Up Auto-Deploy (Optional)

Railway automatically deploys when you push to GitHub!

```bash
# Make changes
git add .
git commit -m "feat: improve predictions"
git push origin master

# Railway automatically deploys!
```

Watch deployment progress in Railway dashboard.

---

## Troubleshooting

### Issue: "Models not loaded"

**Cause**: ML models aren't trained yet

**Solution**: Your app works for the API and Google Sheets integration, but predictions require trained models. For now, the app structure is ready. To train models:

```bash
# Local development
python ml/train_tabular.py
python ml/train_stack.py
```

Then upload models to Railway (see "Deploying Models" section below).

### Issue: "Credentials not configured"

**Cause**: Environment variable not set correctly

**Solution**:
1. Go to Railway dashboard → Variables
2. Ensure `GOOGLE_SHEETS_CREDENTIALS` exists
3. Value should be the entire JSON content
4. Redeploy if needed

### Issue: "Application failed to respond"

**Cause**: Railway needs specific port binding

**Solution**: Check `Procfile` has:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Railway provides `$PORT` automatically.

### Issue: "Spreadsheet not found"

**Solutions**:
1. Verify sheet is shared with service account email
2. Check URL is correct
3. Ensure service account has "Editor" access

---

## Deploying Models (Advanced)

If you want predictions to work, you need to deploy trained models:

### Option 1: Upload to Railway (Small Models)

```bash
# Create models directory in project
mkdir -p models/stacker

# Train models locally
python ml/train_tabular.py

# Models will be saved to models/stacker/

# Commit and push
git add models/stacker/*.joblib
git commit -m "feat: add trained models"
git push origin master
```

### Option 2: Use Cloud Storage (Recommended for Large Models)

1. Upload models to Google Cloud Storage or S3
2. Add download script in Railway startup
3. Set `MODEL_DIR` environment variable

---

## Monitoring & Logs

### View Logs

**Railway Dashboard:**
1. Click on your service
2. Go to "Deployments" tab
3. Click on latest deployment
4. View real-time logs

**CLI:**
```bash
railway logs
```

### Monitor Usage

**Railway Dashboard:**
- "Metrics" tab shows CPU, memory, requests
- "Usage" tab shows monthly costs

---

## Costs

### Railway Pricing

- **Free Tier**: $5/month credit
- **Pro Plan**: $20/month (includes $5 credit)
- **Estimated costs**: ~$2-5/month for light usage

### What Uses Credits

- Compute time (when app is running)
- Memory usage
- Outbound bandwidth

### Optimize Costs

1. **Use sleep mode**: App sleeps when inactive (free tier)
2. **Optimize memory**: Reduce if possible
3. **Batch requests**: Process multiple properties at once

---

## Security Best Practices

1. **Never commit credentials**
   - Add to `.gitignore`
   - Use environment variables only

2. **Restrict service account**
   - Only grant access to specific sheets
   - Use "Editor" not "Owner"

3. **Add authentication** (Optional)
   - Implement API keys
   - Add rate limiting

4. **Monitor access**
   - Check Railway logs regularly
   - Set up alerts for errors

---

## Custom Domain (Optional)

### Add Your Own Domain

1. **Railway Dashboard** → Settings → Domains
2. Click "Add Domain"
3. Enter your domain: `avm.yourdomain.com`
4. Add CNAME record to your DNS:
   - Name: `avm`
   - Value: `your-app.railway.app`

---

## Next Steps

- ✅ Test with real property data
- ✅ Train and deploy ML models
- ✅ Share URL with users
- 🔄 Monitor usage and optimize
- 🔄 Add authentication if needed
- 🔄 Set up custom domain

---

## Support Resources

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **API Docs**: `https://your-app.railway.app/docs`
- **Project Docs**: See `GOOGLE_SHEETS_INTEGRATION.md`

---

## Quick Reference

```bash
# Deploy to Railway
git push origin master

# View logs
railway logs

# Set env var
railway variables set KEY=value

# Open deployed app
railway open
```

Your app URL: `https://your-app.railway.app`

**Congratulations! Your app is live! 🎉**
