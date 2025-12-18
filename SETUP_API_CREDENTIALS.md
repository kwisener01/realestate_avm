# Google Sheets API Setup Guide

Follow these steps to enable automatic Google Sheets updates:

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Name it: `RealEstate-AVM`
4. Click "Create"

## Step 2: Enable Google Sheets API

1. In your project, go to "APIs & Services" → "Library"
2. Search for "Google Sheets API"
3. Click on it and click "Enable"
4. Also search for "Google Drive API" and enable it

## Step 3: Create Service Account Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "Service Account"
3. Fill in:
   - Service account name: `realestate-avm-updater`
   - Service account ID: (auto-filled)
   - Description: `Updates real estate AVM spreadsheets`
4. Click "Create and Continue"
5. Skip the optional steps (click "Continue" → "Done")

## Step 4: Download Credentials JSON

1. Click on the service account you just created
2. Go to the "Keys" tab
3. Click "Add Key" → "Create new key"
4. Select "JSON" format
5. Click "Create"
6. A JSON file will download - save it as:
   ```
   C:\Projects\realestate_avm\credentials.json
   ```

## Step 5: Share Your Google Sheet with Service Account

1. Open the downloaded JSON file
2. Find the line with `"client_email"` - it looks like:
   ```
   realestate-avm-updater@PROJECT-ID.iam.gserviceaccount.com
   ```
3. Copy this email address
4. Open your [Google Sheet](https://docs.google.com/spreadsheets/d/1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM/edit)
5. Click "Share" button
6. Paste the service account email
7. Give it "Editor" permissions
8. Uncheck "Notify people"
9. Click "Share"

## Step 6: Run the Update Script

Once the credentials.json file is in place, run:

```bash
cd scripts
python update_sheet_directly.py
```

The script will automatically add the Deal Status and ARV columns to your sheet!

---

## Quick Checklist:

- [ ] Created Google Cloud project
- [ ] Enabled Google Sheets API
- [ ] Enabled Google Drive API
- [ ] Created service account
- [ ] Downloaded credentials.json to project root
- [ ] Shared Google Sheet with service account email
- [ ] Ran update script

---

**Need help?** Let me know which step you're on and I can provide more details!
