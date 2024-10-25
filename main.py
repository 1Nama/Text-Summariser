import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

class TextSummarizer:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.stopwords = list(STOP_WORDS)
        self.punctuation = punctuation
        # POS tags according to importance order -->
        self.pos_tags = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']

    def summarize_text(self, text, summary_percentage):
        doc = self.nlp(text)
        
        word_frequencies = {}
        # Only considering words that are not stopwords, not punctuation, and belong to specific POS tags
        for word in doc:
            if (word.text.lower() not in self.stopwords 
                and word.text.lower() not in self.punctuation
                and word.pos_ in self.pos_tags):  # POS tagging
                if word.text not in word_frequencies:
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

        # Normalizing the word frequencies 
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency

        # Tokenizing sentences and scoring them based on the word frequencies
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text in word_frequencies.keys():
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word.text]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text]

        new_length = int(len(sentence_tokens) * summary_percentage)
        summary_sentences = nlargest(new_length, sentence_scores, key=sentence_scores.get)

        final_summary = [sent.text for sent in summary_sentences]
        summary = ' '.join(final_summary)

        original_length = len(text.split(' '))
        summary_length = len(summary.split(' '))

        return summary, original_length, summary_length


def main():
    # Input Text
    text = """
Artificial intelligence (AI) has rapidly become one of the most transformative technologies in modern history, impacting nearly every aspect of life, from healthcare and education to transportation and entertainment. At its core, AI involves creating systems that can simulate human intelligence, allowing machines to learn from experience, adapt to new inputs, and perform tasks that once required human intervention. This technology is powered by algorithms and vast amounts of data, enabling machines to recognize patterns, make decisions, and even solve complex problems. In healthcare, AI is revolutionizing the field by improving diagnostics, personalizing treatment plans, and accelerating drug discovery, while in industries like finance, it’s used to detect fraud, optimize trading strategies, and automate customer service. The rise of AI-powered virtual assistants like Siri and Alexa has brought AI into homes, making everyday tasks easier and more efficient. Autonomous vehicles, another major advancement, are poised to reshape transportation by reducing accidents and improving traffic management. However, with all its benefits, AI also presents challenges, such as concerns over privacy, job displacement, and the ethical implications of machines making decisions in areas like law enforcement or warfare. As AI continues to evolve, it holds immense potential to address some of the world's most pressing issues, from climate change to global inequality, but it also requires careful consideration of its societal impact to ensure that it benefits humanity as a whole.
    """

    summarizer = TextSummarizer()
    summary, original_length, summary_length = summarizer.summarize_text(text, 0.5)

    print("Original Length:", original_length)
    print("Summary Length:", summary_length)
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()

