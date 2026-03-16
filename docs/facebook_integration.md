# Facebook Integration — Technical Reference

## Overview

Local Tailor fetches comments from Facebook Page posts via the Graph API. All data stays local — the API is called directly from the user's machine, and comments are saved to `data/comments_clean_{post_id}.json`.

## Prerequisites

1. A **Facebook Page** (not a personal profile) with published posts
2. A **Facebook App** (free, created at developers.facebook.com)
3. A **Page Access Token** with `pages_read_engagement` permission

## Setup Steps

### Step 1: Create a Facebook App

1. Go to [developers.facebook.com/apps](https://developers.facebook.com/apps/)
2. Click **Create App** → choose **Business** type
3. Name it anything (e.g. "Local Tailor")
4. After creation, note the **App ID** (you won't need the App Secret for public page data)

### Step 2: Get a Page Access Token

**Option A: Graph API Explorer (quick, short-lived token)**

1. Go to [developers.facebook.com/tools/explorer](https://developers.facebook.com/tools/explorer/)
2. Select your app from the dropdown
3. Click **Get User Access Token**
4. Under **Permissions**, check:
   - `pages_show_list`
   - `pages_read_engagement`
   - `pages_read_user_content`
5. Click **Generate Access Token** → authorize in the popup
6. In the dropdown next to "User Token", switch to your **Page** → this gives you a **Page Access Token**
7. Copy this token

This token expires in ~1 hour. For longer-lived tokens, see Option B.

**Option B: Long-lived Page Token (recommended for regular use)**

1. Get a short-lived User Access Token from the Explorer (Step A above)
2. Exchange it for a long-lived token:
   ```
   GET https://graph.facebook.com/v21.0/oauth/access_token
     ?grant_type=fb_exchange_token
     &client_id={APP_ID}
     &client_secret={APP_SECRET}
     &fb_exchange_token={SHORT_LIVED_TOKEN}
   ```
   This returns a token valid for ~60 days.
3. Then get the permanent Page Access Token:
   ```
   GET https://graph.facebook.com/v21.0/me/accounts
     ?access_token={LONG_LIVED_USER_TOKEN}
   ```
   The `access_token` field in the response for your page **does not expire**.

### Step 3: Find the Post ID

Every Facebook post has an ID in the format `{page_id}_{post_id}`.

**From a post URL:**
- URL: `https://www.facebook.com/YourPage/posts/1234567890`
- Post ID: `{your_page_id}_1234567890`

**Or list recent posts via API:**
```
GET https://graph.facebook.com/v21.0/{page_id}/posts
  ?access_token={PAGE_TOKEN}
  &fields=id,message,created_time
  &limit=10
```

## Graph API — Fetching Comments

### Endpoint

```
GET https://graph.facebook.com/v21.0/{post_id}/comments
  ?access_token={PAGE_TOKEN}
  &fields=id,message,created_time,like_count,comment_count
  &limit=100
  &order=reverse_chronological
```

### Response Schema

```json
{
  "data": [
    {
      "id": "1234567890_9876543210",
      "message": "love this pillow, so comfortable!",
      "created_time": "2026-03-15T14:30:00+0000",
      "like_count": 3,
      "comment_count": 0
    }
  ],
  "paging": {
    "cursors": {
      "before": "...",
      "after": "..."
    },
    "next": "https://graph.facebook.com/v21.0/..."
  }
}
```

### Pagination

Facebook returns max 100 comments per request. To get all comments:

1. Make the initial request
2. If `paging.next` exists in the response, request that URL
3. Repeat until `paging.next` is absent

```python
import requests

def fetch_all_comments(post_id, access_token):
    url = f"https://graph.facebook.com/v21.0/{post_id}/comments"
    params = {
        "access_token": access_token,
        "fields": "id,message,created_time,like_count,comment_count",
        "limit": 100,
        "order": "reverse_chronological",
    }

    all_comments = []
    while url:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_comments.extend(data.get("data", []))
        url = data.get("paging", {}).get("next")
        params = {}  # next URL already includes params

    return all_comments
```

### Converting to Local Tailor Format

The fetched comments need to be saved in the schema that `pipeline.py` expects:

```python
import json
from datetime import datetime

def save_comments(fb_comments, post_id):
    comments = []
    for i, c in enumerate(fb_comments):
        comments.append({
            "id": c["id"],
            "idx": i,
            "message": c.get("message", ""),
            "created_time": c.get("created_time", ""),
            "like_count": c.get("like_count", 0),
            "comment_count": c.get("comment_count", 0),
        })

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_comments": len(comments),
            "source": "facebook",
            "post_id": post_id,
        },
        "comments": comments,
    }

    path = f"data/comments_clean_{post_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return path
```

### Filtering

Comments to skip before saving:
- Empty `message` field (image-only comments, stickers)
- Messages shorter than 3 characters
- Duplicate messages from the same user (spam)

## Rate Limits

| Tier | Limit | Reset |
|------|-------|-------|
| Standard | 200 calls/user/hour | Rolling window |
| App-level | 200 × active_users/hour | Rolling window |

For a single post with <1000 comments, this is not a concern (10 paginated requests max).

If rate-limited, the API returns HTTP 429 with:
```json
{
  "error": {
    "code": 4,
    "message": "Application request limit reached"
  }
}
```

Handle this with a simple backoff:
```python
import time

def fetch_with_retry(url, params, max_retries=3):
    for attempt in range(max_retries):
        resp = requests.get(url, params=params)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise Exception("Rate limit exceeded after retries")
```

## Token Storage

- **Never commit tokens to git.** The `.gitignore` should include any token file.
- Store the token in a local file: `config/.fb_token` (already gitignored)
- Or use an environment variable: `LOCALTAILOR_FB_TOKEN`
- The Streamlit Login view should save/load from one of these locations

Recommended `.gitignore` additions:
```
config/.fb_token
.env
```

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `OAuthException` (code 190) | Token expired | Re-generate token (see Step 2) |
| `OAuthException` (code 10) | Post not visible to app | Check page permissions |
| HTTP 400 + "Invalid post ID" | Wrong post ID format | Use `{page_id}_{post_id}` format |
| Empty `data` array | Post has no comments, or wrong post | Verify the post URL manually |
| `pages_read_engagement` missing | Permission not granted | Re-authorize with correct permissions |

## End-to-End Flow

```
User pastes access token in Login view
  → token validated against /me API
  → stored in config/.fb_token

User pastes post URL in Facebook view
  → post ID extracted from URL
  → comments fetched via Graph API (paginated)
  → saved to data/comments_clean_{post_id}.json

User runs: python run_pipeline.py predict
  → loads existing models
  → classifies fetched comments
  → saves predictions
  → launches UI with results
```

## API Version

This document uses Graph API **v21.0** (current as of early 2026). Facebook deprecates API versions roughly every 2 years. If calls start failing with version errors, update the version number in the URL (e.g. `v22.0`).

Check current versions at: [developers.facebook.com/docs/graph-api/changelog](https://developers.facebook.com/docs/graph-api/changelog)
