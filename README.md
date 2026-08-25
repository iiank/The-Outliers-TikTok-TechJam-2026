# The-Outliers-TikTok-TechJam-2026

## Development
Before you begin, create a new folder to hold your project.

### Commands to set up connection between Local and Remote Git Repository
```sh
cd insert/your/path/to/the/folder/here
git init
git remote add origin https://github.com/BT3103AppDev1/L1_Group_20.git
git pull origin main
```

### Register Github account
```sh
git config --global user.email "you@example.com"
git config --global user.name "github username"
```

## Before pushing commits (IMPORTANT!)
### Ensure Local Git Repository is in sync with Remote Git Repository
```sh
git branch -m main          # To ensure you're in main branch
git pull origin main        # Pull any recent commits from other developers
```

### Pushing commits
```sh
git add .                                              # To stage recent edits
git commit -m "brief description of commit"            # To commit edits with description
git push 