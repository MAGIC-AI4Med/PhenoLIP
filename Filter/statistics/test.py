import json
with open('data/pmc_json/PMC000xxxxxx/PMC176545.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
captions = []
for figure in data['figure']:
    cap = figure['caption']
    title = cap.get('title', '')
    paragraphs = '\n'.join(cap.get('paragraphs', []))
    captions.append(f"{title}\n{paragraphs}")
text_content = '\n'.join(captions)
print(text_content)