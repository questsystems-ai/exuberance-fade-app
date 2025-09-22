#!/usr/bin/env bash
#
# github_setup_and_push.sh
# Push your current Runpod working tree to a new branch on GitHub.
# Usage:
#   bash github_setup_and_push.sh ssh
#   bash github_setup_and_push.sh https
#
# Notes:
# - For HTTPS you'll need a GitHub Personal Access Token with 'repo' scope.
# - For SSH you'll need to have an SSH key and add the public key to GitHub.
set -euo pipefail

REPO_URL_SSH="git@github.com:questsystems-ai/exuberance-fade-app.git"
REPO_URL_HTTPS="https://github.com/questsystems-ai/exuberance-fade-app.git"
BRANCH="${BRANCH:-gcp-migration}"

if [[ "${1:-}" != "ssh" && "${1:-}" != "https" ]]; then
  echo "Usage: bash $0 [ssh|https]"
  exit 1
fi

# Ensure we're in the repo root (has .git or init it)
if [[ ! -d .git ]]; then
  git init
fi

# Set a robust .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.venv/
venv/
.env
.env.*

# Data & outputs
reports/
logs/
*.log
.ipynb_checkpoints/

# OS/editor
.DS_Store
.idea/
.vscode/
EOF

# Add all & commit
git add -A
git commit -m "GCP migration: snapshot working tree and add .gitignore" || true

# Add remote if missing
if ! git remote get-url origin >/dev/null 2>&1; then
  if [[ "$1" == "ssh" ]]; then
    git remote add origin "$REPO_URL_SSH"
  else
    git remote add origin "$REPO_URL_HTTPS"
  fi
fi

# Create branch and push
git checkout -B "$BRANCH"

if [[ "$1" == "ssh" ]]; then
  # Ensure SSH agent (optional)
  if [[ -f "$HOME/.ssh/id_ed25519" || -f "$HOME/.ssh/id_rsa" ]]; then
    eval "$(ssh-agent -s)"
    ssh-add 2>/dev/null || true
  fi
  git push -u origin "$BRANCH"
else
  echo "Using HTTPS. If prompted for a password, paste a GitHub PAT with 'repo' scope."
  git push -u origin "$BRANCH"
fi

echo "✅ Pushed branch '$BRANCH' to origin."
