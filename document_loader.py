from langchain_community.document_loaders import TextLoader

loader=TextLoader("documents/company_policy.txt",encoding="utf-8")

documents=loader.load()

for document in documents:
    print("Document Content:")
    print(document.page_content)

    print("\nDocument Metadata:")
    print(document.metadata)

