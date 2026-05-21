# Setup Guide

## 1. Install Git from Terminal

Run in PowerShell:

```powershell
winget install --id Git.Git -e --source winget
```

Verify installation:

```powershell
git --version
```

---

## 2. Install Repo from Terminal

Open **Git Bash** (installed with Git), then run:

```bash
mkdir -p ~/bin
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod +x ~/bin/repo
```

Add Repo to PATH:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:

```bash
repo version
```

---

## 3. Set Up Git Name and Email

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Verify configuration:

```bash
git config --global --list
```

---

## 4. Clone Repository

Go to your preferred location:

```bash
cd Desktop
```

Clone the repository:

```bash
git clone https://github.com/gamob/cos30018_group.git
```

---

## 5. Open in VSCode + GitHub Login

Open the cloned folder in VSCode.

When VSCode asks you to sign in to GitHub:

* Click **"Sign in with GitHub"**
* Authorize in browser
* Return to VSCode after successful login

You may not see the login popup immediately. It usually appears when:

* pushing commits
* pulling changes
* syncing branches

---

## 6. Enable Git Auto Fetch in VSCode

In VSCode:

* Press `Ctrl + Shift + P`
* Search: `Git: Enable Auto Fetch`
* Enable it

This automatically checks for remote updates in the background.
