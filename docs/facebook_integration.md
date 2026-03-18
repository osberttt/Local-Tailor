# Facebook Integration

Fetches comments from Facebook Page posts via Graph API. All data stays local, saved to `data/{SHOP}/comments_clean_{post_id}.json`.

## Prerequisites

1. A Facebook **Page** (not personal profile)
2. A Facebook **App** (free, developers.facebook.com)
3. A **Page Access Token** with `pages_read_engagement`

## Token Setup

### Quick (expires ~1 hour)

1. [Graph API Explorer](https://developers.facebook.com/tools/explorer/) → select your app
2. Get User Access Token → check `pages_show_list`, `pages_read_engagement`, `pages_read_user_content`
3. Switch dropdown to your Page → copy the Page Access Token

### Long-lived (recommended)

1. Get short-lived user token (above)
2. Exchange for long-lived token:
   ```
   GET /v21.0/oauth/access_token?grant_type=fb_exchange_token&client_id={APP_ID}&client_secret={APP_SECRET}&fb_exchange_token={SHORT_TOKEN}
   ```
3. Get permanent page token:
   ```
   GET /v21.0/me/accounts?access_token={LONG_LIVED_TOKEN}
   ```

### Storage

Store in `config/.fb_token` (gitignored) or env var `LOCALTAILOR_FB_TOKEN`. Never commit tokens.

## Fetching Comments

### Endpoint

```
GET /v21.0/{post_id}/comments?access_token={TOKEN}&fields=id,message,created_time,like_count,comment_count&limit=100&order=reverse_chronological
```

Post ID format: `{page_id}_{post_id}` (from post URL or `/v21.0/{page_id}/posts`).

### Pagination

Max 100/request. Follow `paging.next` URL until absent.

```python
def fetch_all_comments(post_id, access_token):
    url = f"https://graph.facebook.com/v21.0/{post_id}/comments"
    params = {"access_token": access_token, "fields": "id,message,created_time,like_count,comment_count", "limit": 100}
    all_comments = []
    while url:
        data = requests.get(url, params=params).json()
        all_comments.extend(data.get("data", []))
        url = data.get("paging", {}).get("next")
        params = {}
    return all_comments
```

### Converting to Local Tailor format

```python
output = {
    "metadata": {"generated_at": now, "total_comments": len(comments), "source": "facebook", "post_id": post_id},
    "comments": [{"id": c["id"], "idx": i, "message": c["message"], "created_time": c["created_time"], "like_count": c.get("like_count",0)} for i,c in enumerate(fb_comments)]
}
# Save to: data/{SHOP}/comments_clean_{post_id}.json
```

Filter out: empty messages, <3 chars, duplicate messages from same user.

## Rate Limits

200 calls/user/hour (rolling). Posts with <1000 comments need ~10 requests — not a concern.

On HTTP 429: back off 60s, retry up to 3x.

## Errors

| Error | Fix |
|-------|-----|
| OAuthException code 190 | Token expired — regenerate |
| OAuthException code 10 | Post not visible — check page permissions |
| Invalid post ID | Use `{page_id}_{post_id}` format |
| Empty data array | No comments, or wrong post ID |

## End-to-End Flow

```
Paste token in Login view → validate via /me → store in config/.fb_token
Paste post URL in Facebook view → extract post ID → fetch + paginate → save to data/{SHOP}/
Run: python run_pipeline.py predict → classify → dashboard
```

API version: **v21.0** (2026). Update version number if calls fail with version errors.
