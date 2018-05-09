class Word(object):
    def __init__(self, text, begin, end):
        self.text = text
        self.begin = begin
        self.end = end
        self.opinion = None

    def set_opinion(self, opinion):
        self.opinion = opinion

    def get_polarity(self):
        if self.opinion is None:
            return 0
        else:
            self.opinion.polarity

    def is_colored(self):
        return self.opinion is not None

    def __repr__(self):
        return '<Word "{text}" from {begin} to {end} with opinion {opinion} at {hid}>'.format(
            text = self.text,
            begin = self.begin,
            end = self.end,
            opinion = self.opinion,
            hid = hex(id(self))
        )

class PosTaggedWord(Word):
    def __init__(self, word, pos, tag, vector):
        Word.__init__(self, word.text, word.begin, word.end)
        self.opinion = word.opinion
        self.pos = pos
        self.tag = tag
        self.vector = vector

    def __repr__(self):
        return '<PosTaggedWord "{word}", {pos}#{tag}, {vector} at {hid}>'.format(
            word = self.text,
            pos = self.pos,
            tag = self.tag,
            vector = self.vector,
            hid = hex(id(self))
        )

