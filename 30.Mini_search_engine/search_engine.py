import os
import math
from collections import defaultdict

class SearchEngine:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder
        self.documents = {}
        self.word_freq = defaultdict(lambda: defaultdict(int))
        self.doc_freq = defaultdict(int)
        self.load_documents()
        self.build_index()

    def load_documents(self):
        for filename in os.listdir(self.docs_folder):
            if filename.endswith('.txt'):
                path = os.path.join(self.docs_folder, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    self.documents[filename] = f.read().lower()

    def tokenize(self, text):
        return [word.strip('.,!?;:"()[]') for word in text.lower().split()]

    def build_index(self):
        for doc_name, content in self.documents.items():
            words = self.tokenize(content)
            unique_words = set(words)
            
            for word in words:
                self.word_freq[doc_name][word] += 1
            
            for word in unique_words:
                self.doc_freq[word] += 1

    def tf(self, word, doc_name):
        return self.word_freq[doc_name][word] / len(self.tokenize(self.documents[doc_name]))

    def idf(self, word):
        return math.log(len(self.documents) / (1 + self.doc_freq[word]))

    def tfidf(self, word, doc_name):
        return self.tf(word, doc_name) * self.idf(word)

    def search(self, query):
        query_words = self.tokenize(query)
        scores = defaultdict(float)
        
        for doc_name in self.documents:
            for word in query_words:
                scores[doc_name] += self.tfidf(word, doc_name)
        
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in results if score > 0]

    def display_results(self, results):
        if not results:
            print("\nNo results found.")
            return
        
        print("\n--- Search Results ---")
        for i, (doc_name, score) in enumerate(results, 1):
            print(f"{i}. {doc_name} (Score: {score:.4f})")
            preview = self.documents[doc_name][:100]
            print(f"   Preview: {preview}...\n")

def main():
    docs_folder = os.path.join(os.path.dirname(__file__), 'documents')
    
    if not os.path.exists(docs_folder):
        print("Documents folder not found!")
        return
    
    engine = SearchEngine(docs_folder)
    
    print("=== Mini Search Engine ===")
    print(f"Loaded {len(engine.documents)} documents\n")
    
    while True:
        query = input("Search (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
        
        results = engine.search(query)
        engine.display_results(results)

if __name__ == "__main__":
    main()
