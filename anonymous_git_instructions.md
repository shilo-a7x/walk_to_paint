# Anonymous Code Submission Workflow

Steps to turn `release_staging/` into a view-only, anonymized code link for
double-blind review, without exposing your real identity, GitHub account, or
this machine.

**This file itself must never be committed.** It documents the anonymization
process and names real identifiers (GitHub handle placeholders, review-link
references) that have no business inside the submission. Delete it or move it
out of `release_staging/` before Step 1, not after.

---

## Step 0: Curate contents — before anything is committed

Do this *before* `git init`, not after. Anything that enters git history is
hard to fully remove later (rewriting history + force-push + hoping nothing
cached it in the meantime), so the goal is for the first commit to already be
exactly what should ship.

1. Remove or relocate this file (see above).
2. Confirm `.gitignore` is present and covers `.venv/`, `__pycache__/`,
   `outputs/`, `logs/`, and generated cache files — check it's still there
   and hasn't been edited out.
3. List what a naive `git add .` would actually pick up, and read the list:
   ```bash
   cd release_staging
   git init         # temporary, only to compute the diff — see Step 1's note
                     # about doing this outside the tracked repo instead
   git add -n .      # dry run: prints what WOULD be staged, stages nothing
   ```
   Every path in that output should be something you recognize as meant to
   ship. If anything is a leftover artifact, generated file, or scratch note
   with no `.gitignore` rule covering it, delete it or add a rule now.
4. Re-run the anonymization scan on the actual directory contents (path
   fragments, username, lab name, email, machine-specific paths, internal
   project/paper codenames — whatever applies to your setup):
   ```bash
   grep -rniE "<your-real-name-or-handle>|<lab-or-org-name>|/home/|@<your-email-domain>|<internal-codename>" .
   ```
   This should return nothing. If it does, fix the source file, not just the
   grep target.

---

## Step 1: Isolate from the existing repo, then set an anonymous git identity

`release_staging/` currently lives as plain files inside an already
git-tracked repository (no `.git` of its own). Running `git init` directly
inside it nests a new repo inside the tracked one, which git will treat as an
embedded-repo/gitlink entry in the outer repo — confusing, and easy to
mishandle (e.g. accidentally committing a gitlink stub upstream instead of
the real files). Avoid this entirely:

```bash
# Copy the curated contents somewhere fully outside the existing repo:
cp -r /path/to/walk_to_paint/release_staging /path/to/somewhere-else/my-anon-repo
cd /path/to/somewhere-else/my-anon-repo
```

Then set an identity scoped to this folder only (no `--global` flag, so it
never touches your real git config elsewhere):

```bash
git config user.name "Anonymous Author"
git config user.email "anon@example.com"
```

---

## Step 2: Initialize and commit

```bash
git init
git add .
git status --short        # read this before committing — same check as Step 0.3,
                           # now against the real repo instead of a dry run
git commit -m "Initial commit"
```

**Final gate, immediately after commit, before anything is pushed anywhere:**

```bash
git show --stat HEAD                          # every file that just got committed
git log --format='%an <%ae>' -1                # confirm the anonymous identity took
git grep -niE "<your-real-name-or-handle>|<lab-or-org-name>|/home/|@<your-email-domain>|<internal-codename>"
```
The last command should return nothing. This is the last checkpoint before
the content becomes effectively public — treat a hit here as a hard stop, not
something to patch after pushing.

---

## Step 3: Push to GitHub

The repo will be associated with whichever GitHub account you push it to —
the identity you set in Step 1 only affects *commit metadata* (author name/
email on each commit), not which account owns the repo on GitHub itself. Two
real options, with a real tradeoff:

- **Push to your existing account, repo set to Private.** Lower effort. The
  masking tool in Step 4 requests access to private repos through its GitHub
  App, so this works. As long as the repo stays Private, it isn't visible on
  your public profile or discoverable by anyone without the anonymized link.
- **Push to a fresh, throwaway GitHub account** (no real name, no linked
  email) if you want the repo to be Public, or if you don't want to trust the
  masking tool's correctness as your only safeguard.

**Avoid: pushing to your existing account with the repo set to Public.** The
masking tool only hides your identity *within its own wrapped view* — the
underlying GitHub repo is still fully attributed to your real account and
visible to anyone who reaches it directly (search, your profile's repo list,
anyone who independently learns your username). This is the one combination
that defeats the entire point of this workflow.

```bash
git remote add origin https://github.com/<account>/<new-repo-name>.git
git branch -M main
git push -u origin main
```

---

## Step 4: Generate the masked anonymous link

1. Go to [anonymous.4open.science](https://anonymous.4open.science/).
2. Log in and authorize it via GitHub (this authorization itself is tied to
   whichever account you push from in Step 3 — an accepted, unavoidable part
   of using this class of tool).
3. **Create a new anonymous repository** → select the repo from Step 3.
4. Configure anonymity rules:
   - **Anonymize profile names** — mask your GitHub username in the wrapper view.
   - **Anonymize dates** — hide commit/creation timestamps in the wrapper view.
   - **Anonymization tokens** — list your real name, handle, and institution;
     the tool overwrites exact matches with `XXXX` in the rendered view.
5. Generate the link and test it yourself in a private/incognito browser
   window before sending it anywhere, to confirm it renders the way you expect.

---

## What this does and doesn't protect against

- The tokens/date/profile masking in Step 4 only apply to the
  **anonymous.4open.science rendering** of the repo — they do not change the
  actual GitHub repo, its commit metadata, or its visibility settings. Step 3
  is what actually controls whether the real repo is reachable outside that
  wrapper.
- **No `git clone`:** reviewers interact with the wrapper's web UI and a
  "Download ZIP" button, not a real git remote.
- **Isolated view:** reviewers see only this project through the wrapper —
  not your account, other repos, or activity — but only for as long as they
  stay inside the wrapped link rather than following any link back to the
  real GitHub repo.
