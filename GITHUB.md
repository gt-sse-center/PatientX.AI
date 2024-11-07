# Getting Started with GitHub

Welcome to GitHub! This guide will help you get started using GitHub, a platform for version control and collaboration. If you’re new to version control, GitHub is a great tool to manage and track changes to your code or files and work with others.

## What You’ll Learn

1. What GitHub is and why it’s useful.
2. How to create a GitHub account.
3. Basic Git commands to work with GitHub.
4. Creating your first repository.
5. Making and tracking changes.

---

## 1. What is GitHub?

**GitHub** is a platform built around **Git**, a version control system that helps you manage changes to your codebase or documents. GitHub allows multiple people to work on a project at the same time and keeps track of each change.

### Why Use GitHub?

- **Version Tracking**: Easily keep track of what changes were made and who made them.
- **Backup**: Store your work safely in the cloud.
- **Collaboration**: Work on projects with others efficiently.

---

## 2. Set Up Your GitHub Account

1. Go to [GitHub.com](https://github.com/).
2. Click **Sign up** and create a free account.
3. Complete the sign-up steps, and you’re ready to go!

---

## 3. Basic Git Commands

Before using GitHub, you’ll need to install **Git** (if you haven’t yet):

- [Install Git](https://git-scm.com/downloads)

### Git Commands Cheat Sheet

| Command                       | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `git init`                    | Initializes a new Git repository in your folder            |
| `git clone <url>`             | Downloads a project from GitHub to your computer           |
| `git add <file>`              | Adds a file to the list of changes to be saved             |
| `git commit -m "message"`     | Saves your changes with a description of what you did      |
| `git push`                    | Sends your changes to GitHub                               |
| `git pull`                    | Updates your local files with changes from GitHub          |

---

## 4. Cloning a repository

Cloning a repository means downloading a copy of an existing project from GitHub to your local computer. This allows you to work on the project locally and keep it synchronized with the version on GitHub.

### Steps to Clone a Repository

1. **Find the repository you want to clone** on GitHub.
2. Click the **Code** button on the repository's main page.
3. Copy the **URL** (you can choose HTTPS, SSH, or GitHub CLI, but HTTPS is usually easiest for beginners).

   - For HTTPS, the URL will look something like this:
     ```
     https://github.com/username/repository-name.git
     ```

4. Open a terminal (or Git Bash on Windows).
5. Navigate to the directory where you want to save the repository:
   ```bash
   cd path/to/your/directory
   ```
6. Use the git clone command, followed by the repository URL:
    ```bash
    git clone https://github.com/username/repository-name.git
    ```

7. 	Git will download the repository to your local machine. You can now enter the repository’s directory:
    ```bash
    cd repository-name 
    ```
---

## 5. Making Your First Commit

A **commit** is a snapshot of your changes.

1. After creating or cloning your repository and entering the project directory, you can create/add/modify any files
2. Tell git to begin tracking this file with ``` git add <filename>```
   or
    ```git add .```
   to track the entire directory
3. Commit your changes with a message
    ```bash
    git commit -m "<message>"
    ```
4. Push changes to GitHub
    ```bash 
    git push
    ```
---

## 6. Pulling changes
If you’re working with others, you’ll want to keep your files up-to-date with changes they make.

1.	Use git pull to fetch and merge changes from GitHub:
```bash
git pull 
```
---

## 7. Basic Recap

1.	**Add:** Track files with ```git add <file>```
2. **Commit:** Commit files with ```git commit -m "<message>"```
3. **Push:** Send your changes to GitHub with ```git push```


