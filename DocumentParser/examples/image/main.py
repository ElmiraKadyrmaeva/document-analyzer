from document_parser import DocumentParser

document_parser = DocumentParser()
result = document_parser.parse('image.png')

print(result)
with open('result.json', 'w', encoding='utf-8') as file:
    file.write(result.to_json(indent=4))
