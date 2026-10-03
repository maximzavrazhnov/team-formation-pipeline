# How to publish this project on GitHub

## Option A — easiest: upload through the GitHub website

1. Create a GitHub account at github.com if you do not have one.
2. Click **New repository**.
3. Repository name: `team-formation-pipeline`.
4. Description: `Research prototype for project-team formation using digital traces and cooperative-game-inspired methods.`
5. Choose **Public** if you want to use the repository as a portfolio link.
6. Do **not** add another README, .gitignore, or license during repository creation — this package already contains README and .gitignore.
7. Create the repository.
8. Open **Add file → Upload files**.
9. Upload the **contents of this folder**, not the ZIP itself.
10. Commit message: `Initial public research prototype`.
11. Click **Commit changes**.

After upload, check that the root of the repository contains `README.md`, `src/`, `data/`, `docs/`, `notebooks/`, and `scripts/`.

## Option B — recommended long-term: Git from your computer

Install Git, then in a terminal opened inside this project folder run:

```bash
git init
git add .
git commit -m "Initial public research prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/team-formation-pipeline.git
git push -u origin main
```

For future changes:

```bash
git status
git add .
git commit -m "Describe what changed"
git push
```

## Before every push

Run `git status` and verify that you are **not** publishing real Jira exports, employee IDs, credentials, tokens, API keys, or private research data. `.gitignore` blocks common private/generated files, but you still need to review the staged files.

## Suggested GitHub profile bio

`PhD student @ SPbPU | Applied Math & Systems Analysis | Python / C++ / SQL | Data Analysis, Cooperative Game Theory, AI Agents & Cybersecurity`

## Suggested repository topics

`python`, `data-analysis`, `optimization`, `cooperative-game-theory`, `shapley-value`, `team-formation`, `research`, `jira`, `monte-carlo`
