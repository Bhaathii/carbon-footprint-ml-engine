# Push Tourism Carbon Footprint Calculator to GitHub

## Step 1: Create a New Repository on GitHub

1. Go to https://github.com/new
2. Sign in with your GitHub account (Bhaathii)
3. Fill in the details:
   - **Repository name:** tourism-carbon-footprint-calculator
   - **Description:** A web application to calculate and predict tourism trip carbon emissions using machine learning
   - **Public/Private:** Public (recommended for portfolio)
   - **Initialize with:** Skip (we already have files)
4. Click "Create repository"

---

## Step 2: Initialize Git in Your Project Folder

Open PowerShell in your project folder and run:

```powershell
cd C:\Users\ASUS\Desktop\new

# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Tourism Carbon Footprint Calculator with ML model trained on 5000 samples"
```

---

## Step 3: Add Remote Repository

Copy the URL from your GitHub repo (should look like):
`https://github.com/Bhaathii/tourism-carbon-footprint-calculator.git`

Then run:

```powershell
# Add remote origin (replace URL with your repo URL)
git remote add origin https://github.com/Bhaathii/tourism-carbon-footprint-calculator.git

# Verify remote was added
git remote -v
```

---

## Step 4: Push to GitHub

```powershell
# Rename branch to main (GitHub default)
git branch -M main

# Push to GitHub
git push -u origin main
```

You may be prompted for authentication. Use your GitHub token:
- Personal Access Token (recommended)
- Or GitHub CLI authentication

---

## Step 5: Create .gitignore File

Create a `.gitignore` file to exclude unnecessary files:

```
__pycache__/
*.pyc
.streamlit/
.env
*.pkl
*.csv
.DS_Store
venv/
*.egg-info/
.pytest_cache/
```

Then commit:

```powershell
git add .gitignore
git commit -m "Add .gitignore file"
git push
```

---

## Step 6: Create README.md

The project report already has good content. Create a `README.md`:

```powershell
# Copy your project report as README
Copy-Item "PROJECT_REPORT.md" "README.md"

git add README.md
git commit -m "Add comprehensive README and project documentation"
git push
```

---

## Step 7: Create requirements.txt

List all Python dependencies:

```powershell
# Generate requirements file
pip freeze > requirements.txt

git add requirements.txt
git commit -m "Add Python dependencies"
git push
```

---

## Complete PowerShell Commands (Copy & Paste):

```powershell
# Navigate to project
cd C:\Users\ASUS\Desktop\new

# Initialize git
git init
git add .
git commit -m "Initial commit: Tourism Carbon Footprint Calculator"

# Add remote (REPLACE URL WITH YOUR REPO URL)
git remote add origin https://github.com/Bhaathii/tourism-carbon-footprint-calculator.git
git branch -M main
git push -u origin main

# Create and push .gitignore
@'
__pycache__/
*.pyc
.streamlit/
.env
model/model.pkl
model/features.pkl
data/
.DS_Store
venv/
*.egg-info/
.pytest_cache/
'@ | Out-File .gitignore -Encoding UTF8

git add .gitignore
git commit -m "Add .gitignore"
git push

# Copy README
Copy-Item "PROJECT_REPORT.md" "README.md" -Force
git add README.md
git commit -m "Add README"
git push

# Generate and push requirements
pip freeze | Out-File requirements.txt -Encoding UTF8
git add requirements.txt
git commit -m "Add Python dependencies"
git push
```

---

## Folder Structure to Push:

```
tourism-carbon-footprint-calculator/
├── code/
│   ├── app.py                 (Web interface)
│   ├── training.py            (Model training)
│   ├── predict.py             (Prediction script)
│   ├── recomendation.py       (Recommendations)
│   └── __pycache__/          (ignore this)
├── data/
│   └── tourism_5000_rows.csv (training data)
├── model/
│   ├── model.pkl             (trained model)
│   └── features.pkl          (feature list)
├── PROJECT_REPORT.md
├── README.md                 (copy of report)
├── requirements.txt
├── .gitignore
└── GITHUB_SETUP.md           (this file)
```

---

## GitHub Repository Features to Add Later:

1. **Add Topics:** machine-learning, carbon-emissions, tourism, streamlit
2. **Add License:** MIT License (optional but recommended)
3. **Enable Issues:** For bug tracking
4. **Enable Discussions:** For community engagement
5. **Add CI/CD:** GitHub Actions (optional)

---

## Verification Checklist:

- [ ] Repository created on GitHub
- [ ] Files pushed to main branch
- [ ] .gitignore working (no unnecessary files)
- [ ] README.md displays properly
- [ ] requirements.txt lists all dependencies
- [ ] All code files are present
- [ ] Model files (.pkl) are uploaded
- [ ] Project report is in repository

---

## If You Get Authentication Error:

GitHub no longer accepts password authentication. Use one of:

**Option 1: GitHub CLI (Recommended)**
```powershell
# Install GitHub CLI
choco install gh

# Authenticate
gh auth login

# Then retry push
git push
```

**Option 2: Personal Access Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Create new token with repo scope
3. Use token as password when prompted

**Option 3: SSH Keys**
1. Generate SSH key
2. Add to GitHub SSH settings
3. Use SSH URL instead of HTTPS

---

## After Pushing - Showcase Your Work:

1. Add link to your GitHub profile
2. Update your portfolio/resume
3. Share the repository on LinkedIn
4. Add "See on GitHub" link to your project report

---

## Commands Quick Reference:

```powershell
# Check status
git status

# View commit history
git log --oneline

# See what changed
git diff

# Undo last commit (if needed)
git reset --soft HEAD~1

# Push updates
git push origin main
```

---

**You're all set! Push your amazing project to GitHub and share it with the world!** 🚀

