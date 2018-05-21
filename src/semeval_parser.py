import xml.etree.ElementTree as ET
from nltk.tokenize import WordPunctTokenizer
from src.parser import Word, Dataset


class Opinion(object):
    def __init__(self, begin=0, end=0, target='', polarity=1, cat_first='', cat_second=''):
        self.polarity_values = {
            'positive': 3,
            'neutral': 1,
            'negative': 0,
            'conflict': 2
        }
        self.rev_polarity_values = {value: key for key, value in self.polarity_values.items()}

        self.begin = begin
        self.end = end
        self.target = target
        self.polarity = polarity
        self.cat_first = cat_first
        self.cat_second = cat_second
        self.words = []

    def parse(self, node):
        self.begin = int(node.get('from'))
        self.end = int(node.get('to'))
        self.target = node.get('target')
        self.polarity = self.polarity_values[node.get('polarity')]
        category = node.get('category')
        self.cat_first = category.split('#')[0]
        self.cat_second = category.split('#')[1]

    def __repr__(self):
        return "<Opinion {begin}:{end} {c1}#{c2} {polarity} at {hid}>".format(
            begin=self.begin,
            end=self.end,
            c1=self.cat_first,
            c2=self.cat_second,
            polarity=self.polarity,
            hid=hex(id(self))
        )

    def is_empty(self):
        return self.target == ""

    def inflate_target(self):
        self.target = " ".join([word.text for word in self.words]).replace('"', "'").replace('&', '#')

    def to_xml(self):
        return '<Opinion target="{target}" category="{category}" polarity="{polarity}" from="{begin}" to="{end}"/>\n'.format(
            begin=self.begin, end=self.end, target=self.target,
            category=self.cat_first + "#" + self.cat_second,
            polarity=self.rev_polarity_values[self.polarity])


class Sentence:
    def __init__(self, sid="", text=""):
        self.text = text
        self.sid = sid
        self.aspects = []

    def parse(self, node):
        self.text = node.find(".//text").text
        self.sid = node.get("id")
        for opinion_node in node.findall(".//Opinion"):
            opinion = Opinion()
            opinion.parse(opinion_node)
            self.aspects.append(opinion)

    def to_xml(self):
        opinions_xml = '<Opinions>\n'
        for opinion in self.aspects:
            opinions_xml += opinion.to_xml()
        opinions_xml += '</Opinions>\n'
        if not self.aspects:
            opinions_xml = '<Opinions/>'
        return '<sentence id="{sid}">\n<text>{text}</text>\n{opinions}</sentence>\n'.format(
            sid=self.sid,
            text=self.text.replace("&", "#"),
            opinions=opinions_xml)


class Review(object):
    def __init__(self, rid=''):
        self.rid = rid
        self.sentences = []
        self.parsed_sentences = []
        self.aspects = []

    def parse(self, node):
        self.rid = node.get('rid')
        for sentence_node in node.findall(".//sentence"):
            sentence = Sentence()
            sentence.parse(sentence_node)
            self.parsed_sentences.append(sentence)
            self.aspects += self.parsed_sentences[-1].aspects

    def to_xml(self):
        sentences_xml = ''
        for sentence in self.parsed_sentences:
            sentences_xml += sentence.to_xml()
        return '<Review rid="{rid}">\n<sentences>\n{sentences}</sentences>\n</Review>\n'.format(
            rid=self.rid,
            sentences=sentences_xml)


class SemEvalDataset(Dataset):
    def __init__(self, language):
        super().__init__()
        self.language = language

    def parse(self, filename):
        assert filename.endswith('xml')
        tree = ET.parse(filename)
        root = tree.getroot()
        self.reviews = []
        for review_node in root.findall(".//Review"):
            review = Review()
            review.parse(review_node)
            self.reviews.append(review)
        self.tokenize()
        self.pos_tag()

    def tokenize(self):
        for review in self.reviews:
            for i, sentence in enumerate(review.parsed_sentences):
                text = sentence.text
                words_borders = list(WordPunctTokenizer().span_tokenize(text))
                tokenized_sentence = []
                for word_begin, word_end in words_borders:
                    word_text = text[word_begin: word_end]
                    word = Word(word_text, word_begin, word_end)
                    word.set_sid(sentence.sid)
                    for opinion in sentence.aspects:
                        if opinion.target == 'NULL' or opinion.begin == 0 and opinion.end == 0:
                            continue
                        if word.begin >= opinion.begin and word.end <= opinion.end:
                            word.add_opinion(opinion)
                            opinion.words.append(word)
                    tokenized_sentence.append(word)
                review.sentences.append(tokenized_sentence)

    def get_aspect_categories(self):
        categories = set()
        for review in self.reviews:
            for aspect in review.aspects:
                categories.add(aspect.cat_first+"#"+aspect.cat_second)
        categories = list(sorted(list(categories)))
        return {category: i for i, category in enumerate(categories)}