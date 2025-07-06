import feedparser
import json
import time

# Number of entries per category 
papers_per_category = 100

# Category code in human-readable name
categories = {
    "cs.RO": "Robotics",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision",
    "cs.DM": "Discrete Mathematics"
}

def get_data(categories, papers_per_category):
    # Base API URL
    base_url = "http://export.arxiv.org/api/query?"

    all_entries = []

    # Loop over each category
    for cat_code, cat_name in categories.items():
        collected = 0
        start = 0
        batch_size = 100  # arXiv API limit per call
        print(f"Fetching {cat_name}...")

        while collected < papers_per_category:
            query = (f"search_query=cat:{cat_code}&start={start}&max_results={batch_size}"
                    f"&sortBy=submittedDate&sortOrder=descending")
            feed = feedparser.parse(base_url + query)

            if not feed.entries:
                print(f"No more results for {cat_name}. Got {collected}.")
                break

            for entry in feed.entries:
                if collected >= papers_per_category:
                    break
                entry_data = {
                    'title': entry.get('title'),
                    'id': entry.get('id'),
                    'published': entry.get('published'),
                    'updated': entry.get('updated'),
                    'summary': entry.get('summary'),
                    'authors': [author.name for author in entry.get('authors', [])],
                    'primary_category': entry.get('arxiv_primary_category', {}).get('term'),
                    'categories': [tag['term'] for tag in entry.get('tags', [])],
                    'pdf_url': next((link.href for link in entry.links if link.type == 'application/pdf'), None),
                    'comment': entry.get('arxiv_comment'),
                    'journal_ref': entry.get('arxiv_journal_ref'),
                    'category_name': cat_name  # Add human-readable name
                }
                all_entries.append(entry_data)
                collected += 1

            start += batch_size
            time.sleep(1)  # Respect arXiv rate limits

    # Save to JSON
    with open("arxiv_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_entries)} entries to arxiv_dataset.json")


if __name__ == "__main__":
    get_data(categories, papers_per_category)