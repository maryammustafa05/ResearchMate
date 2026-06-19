import requests
import feedparser
def search_arxiv(query,max_results=3):
    base_url="http://export.arxiv.org/api/query"
    params={
        "search_query":f"all:{query}",
        "start":0,
        "max_results":max_results
    }
    response=requests.get(base_url,params=params)
    feed=feedparser.parse(response.text)
    papers=[]
    for entry in feed.entries:
        pdf_link=None
        for link in entry.links:
            if link.type=="application/pdf":
                pdf_link=link.href
                break
        papers.append({
            "title": entry.title,
            "summary": entry.summary,
            "pdf_url": pdf_link
        })
    return papers
def download_pdf(pdf_url):
    response=requests.get(pdf_url)
    return response.content

if __name__ == "__main__":
    from main import read_pdf_bytes
    results = search_arxiv("sign language recognition deep learning")
    first_paper=results[0]
    print(f"Downloading : {first_paper['title']}")
    pdf_bytes=download_pdf(first_paper['pdf_url'])
    print("Extracting text...")
    text = read_pdf_bytes(pdf_bytes)
    print(f"\nExtracted {len(text)} characters")
    print(f"\nFirst 300 characters:\n{text[:300]}")